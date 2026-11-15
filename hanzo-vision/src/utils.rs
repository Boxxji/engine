use std::borrow::Cow;

use fast_image_resize::images::{Image as FirImage, ImageRef as FirImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use hanzo_ml::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView, RgbImage};

fn planar_chw_f32(
    raw: &[u8],
    w: usize,
    h: usize,
    n_channels: usize,
    scales: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    let plane = w * h;
    let mut out = vec![0.0f32; n_channels * plane];
    match n_channels {
        1 => {
            let s = scales[0];
            let b = biases[0];
            for (dst, &px) in out.iter_mut().zip(raw.iter()) {
                *dst = f32::from(px) * s + b;
            }
        }
        3 => {
            let (p0, rest) = out.split_at_mut(plane);
            let (p1, p2) = rest.split_at_mut(plane);
            let (s0, s1, s2) = (scales[0], scales[1], scales[2]);
            let (b0, b1, b2) = (biases[0], biases[1], biases[2]);
            for (idx, &[c0, c1, c2]) in raw.as_chunks::<3>().0.iter().enumerate() {
                p0[idx] = f32::from(c0) * s0 + b0;
                p1[idx] = f32::from(c1) * s1 + b1;
                p2[idx] = f32::from(c2) * s2 + b2;
            }
        }
        _ => {
            for (idx, chunk) in raw.chunks_exact(n_channels).enumerate() {
                for c in 0..n_channels {
                    out[c * plane + idx] = f32::from(chunk[c]) * scales[c] + biases[c];
                }
            }
        }
    }
    out
}

fn image_to_planar_tensor(
    image: &DynamicImage,
    scales: &[f32; 4],
    biases: &[f32; 4],
    device: &Device,
) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let n_channels = n_channels(image);
    let raw: Cow<'_, [u8]> = match n_channels {
        1 => match image.as_luma8() {
            Some(img) => Cow::Borrowed(img.as_raw()),
            None => Cow::Owned(image.to_luma8().into_raw()),
        },
        2 => match image.as_luma_alpha8() {
            Some(img) => Cow::Borrowed(img.as_raw()),
            None => Cow::Owned(image.to_luma_alpha8().into_raw()),
        },
        3 => match image.as_rgb8() {
            Some(img) => Cow::Borrowed(img.as_raw()),
            None => Cow::Owned(image.to_rgb8().into_raw()),
        },
        4 => match image.as_rgba8() {
            Some(img) => Cow::Borrowed(img.as_raw()),
            None => Cow::Owned(image.to_rgba8().into_raw()),
        },
        _ => hanzo_ml::bail!("Unsupported channel count {n_channels}"),
    };
    let out = planar_chw_f32(&raw, w as usize, h as usize, n_channels, scales, biases);
    Tensor::from_vec(out, (n_channels, h as usize, w as usize), device)
}

/// Output is (c, h, w) in `[0.0, 255.0]`.
pub(crate) fn image_to_pixels(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    image_to_planar_tensor(image, &[1.0; 4], &[0.0; 4], device)
}

/// Output is (c, h, w) scaled into `[0.0, 1.0]` in a single pass.
pub(crate) fn image_to_unit_pixels(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    const INV_255: f32 = 1.0 / 255.0;
    image_to_planar_tensor(image, &[INV_255; 4], &[0.0; 4], device)
}

/// Convert `image` to RGB8, scale to `[0, 1]`, normalize per-channel with `(x - mean) / std`,
/// and output an `F32` `(3, h, w)` tensor in a single planar CHW pass.
pub fn image_to_normalized_pixels(
    image: &DynamicImage,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let scales = [
        1.0f32 / (255.0 * std[0]),
        1.0f32 / (255.0 * std[1]),
        1.0f32 / (255.0 * std[2]),
    ];
    let biases = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let rgb: Cow<'_, RgbImage> = match image.as_rgb8() {
        Some(img) => Cow::Borrowed(img),
        None => Cow::Owned(image.to_rgb8()),
    };
    let out = planar_chw_f32(rgb.as_raw(), w as usize, h as usize, 3, &scales, &biases);
    Tensor::from_vec(out, (3, h as usize, w as usize), device)
}

/// Resize an image to `dst_w`x`dst_h` RGB8 using SIMD-accelerated Catmull-Rom resampling.
pub fn simd_resize_rgb(image: &DynamicImage, dst_w: u32, dst_h: u32) -> Result<DynamicImage> {
    if dst_w == 0 || dst_h == 0 {
        hanzo_ml::bail!("invalid resize target dimensions {dst_w}x{dst_h}");
    }
    let rgb: Cow<'_, RgbImage> = match image.as_rgb8() {
        Some(img) => Cow::Borrowed(img),
        None => Cow::Owned(image.to_rgb8()),
    };
    let (src_w, src_h) = (rgb.width(), rgb.height());
    if src_w == 0 || src_h == 0 {
        hanzo_ml::bail!("cannot resize empty image {src_w}x{src_h}");
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(DynamicImage::ImageRgb8(rgb.into_owned()));
    }

    let src_view = FirImageRef::new(src_w, src_h, rgb.as_raw(), PixelType::U8x3)
        .map_err(|e| hanzo_ml::Error::Msg(e.to_string()))?;
    let mut dst_view = FirImage::new(dst_w, dst_h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    resizer
        .resize(&src_view, &mut dst_view, Some(&options))
        .map_err(|e| hanzo_ml::Error::Msg(e.to_string()))?;

    let resized = RgbImage::from_raw(dst_w, dst_h, dst_view.into_vec())
        .ok_or_else(|| hanzo_ml::Error::Msg("resized RGB buffer size mismatch".into()))?;
    Ok(DynamicImage::ImageRgb8(resized))
}

pub(crate) fn n_channels(image: &DynamicImage) -> usize {
    image.color().channel_count() as usize
}
