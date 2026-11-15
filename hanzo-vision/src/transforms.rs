use crate::utils::{image_to_normalized_pixels, image_to_pixels, image_to_unit_pixels};
use hanzo_ml::{Device, Result, Tensor, D};
use image::DynamicImage;

use crate::ImageTransform;

/// Convert an image to a tensor. This converts the data from being in `[0, 255]` to `[0.0, 1.0]`.
/// The tensor's shape is (channels, height, width).
pub struct ToTensor;

impl ImageTransform for ToTensor {
    type Input = DynamicImage;
    type Output = Tensor;
    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        image_to_unit_pixels(x, device)
    }
}

/// Fused RGB8 -> `[0, 1]` -> per-channel `(x - mean) / std` -> `(3, height, width)` `F32` tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToNormalizedTensor {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageTransform for ToNormalizedTensor {
    type Input = DynamicImage;
    type Output = Tensor;

    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        image_to_normalized_pixels(x, &self.mean, &self.std, device)
    }
}

/// Convert an image to a tensor without normalizing to `[0.0, 1.0]`.
/// The tensor's shape is (channels, height, width).
pub struct ToTensorNoNorm;

impl ImageTransform for ToTensorNoNorm {
    type Input = DynamicImage;
    type Output = Tensor;
    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        image_to_pixels(x, device)
    }
}

/// Normalize the image data based on the mean and standard deviation.
/// The value is computed as follows:
/// `
/// x[channel] = (x[channel] - mean[channel]) / std[channel]
/// `
///
/// Expects an input tensor of shape (channels, height, width).
pub struct Normalize {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl ImageTransform for Normalize {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        let num_channels = x.dim(D::Minus(3))?;
        if self.mean.len() != num_channels || self.std.len() != num_channels {
            hanzo_ml::bail!(
                "Num channels ({}) must match number of mean ({}) and std ({}).",
                num_channels,
                self.mean.len(),
                self.std.len()
            );
        }
        let dtype = x.dtype();
        let device = x.device();
        let mean = Tensor::from_slice(
            &self.mean.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            (num_channels,),
            device,
        )?
        .to_dtype(dtype)?;
        let std = Tensor::from_slice(
            &self.std.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            (num_channels,),
            device,
        )?
        .to_dtype(dtype)?;
        let mean = mean.reshape((num_channels, 1, 1))?;
        let std = std.reshape((num_channels, 1, 1))?;
        x.broadcast_sub(&mean)?.broadcast_div(&std)
    }
}

/// Resize the image via nearest interpolation.
pub struct InterpolateResize {
    pub target_w: usize,
    pub target_h: usize,
}

impl ImageTransform for InterpolateResize {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        x.unsqueeze(0)?
            .interpolate2d(self.target_h, self.target_w)?
            .squeeze(0)
    }
}

impl<T: ImageTransform<Input = E, Output = E>, E: Clone> ImageTransform for Option<T> {
    type Input = T::Input;
    type Output = T::Output;

    fn map(&self, x: &T::Input, dev: &Device) -> Result<T::Output> {
        if let Some(this) = self {
            this.map(x, dev)
        } else {
            Ok(x.clone())
        }
    }
}

/// Multiply the pixe values by the provided factor.
///
/// Each pixel value is calculated as follows: x = x * factor
pub struct Rescale {
    pub factor: Option<f64>,
}

impl ImageTransform for Rescale {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        if let Some(factor) = self.factor {
            x * factor
        } else {
            Ok(x.clone())
        }
    }
}

mod tests {
    #[test]
    fn test_to_tensor() {
        use hanzo_ml::Device;
        use image::{ColorType, DynamicImage};

        use crate::ImageTransform;

        use super::ToTensor;

        let image = DynamicImage::new(4, 5, ColorType::Rgb8);
        let res = ToTensor.map(&image, &Device::Cpu).unwrap();
        assert_eq!(res.dims(), &[3, 5, 4])
    }

    #[test]
    fn test_normalize() {
        use crate::{ImageTransform, Normalize};
        use hanzo_ml::{Device, Tensor};

        let image = Tensor::randn(1f32, 0f32, (3, 5, 4), &Device::Cpu).unwrap();
        let res = Normalize {
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        }
        .map(&image, &Device::Cpu)
        .unwrap();
        assert_eq!(res.dims(), &[3, 5, 4])
    }

    #[test]
    fn test_to_normalized_tensor_and_simd_resize_parity() {
        use crate::{simd_resize_rgb, ImageTransform, Normalize, ToNormalizedTensor, ToTensor};
        use hanzo_ml::Device;
        use image::{DynamicImage, Rgb, RgbImage};

        let rgb = RgbImage::from_fn(29, 17, |x, y| {
            Rgb([
                ((x * 37 + y * 13) % 256) as u8,
                ((x * 11 + y * 59) % 256) as u8,
                ((x * 73 + y * 23) % 256) as u8,
            ])
        });
        let img = DynamicImage::ImageRgb8(rgb);
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std = [0.26862954f32, 0.26130258, 0.27577711];

        let unit = ToTensor.map(&img, &Device::Cpu).unwrap();
        let reference = Normalize {
            mean: mean.iter().map(|&v| v as f64).collect(),
            std: std.iter().map(|&v| v as f64).collect(),
        }
        .map(&unit, &Device::Cpu)
        .unwrap();
        let fused = ToNormalizedTensor { mean, std }.map(&img, &Device::Cpu).unwrap();

        assert_eq!(fused.dims(), reference.dims());
        let diff = (fused - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "max abs diff {diff} exceeded 1e-6");

        let resized = simd_resize_rgb(&img, 42, 24).unwrap();
        assert_eq!(resized.width(), 42);
        assert_eq!(resized.height(), 24);
    }
}
