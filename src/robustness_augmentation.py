"""ENHANCEMENT-04: Robustness Module with Augmentation.

Augmentation techniques to improve detection of degraded videos:
  - Compression artifact simulation
  - Blur simulation
  - Brightness variation
  - Noise injection
  - Scaling and rotation
  
Used during both training and inference (test-time augmentation).
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from .utils import setup_logger

logger = setup_logger(__name__)


class RobustnessAugmentation:
    """Augmentation for robustness to video degradation.
    
    Improves detection of:
    - Compressed videos (JPEG artifacts)
    - Blurry frames (motion blur, focus blur)
    - Low-light videos (brightness variations)
    - Side angles (scale/rotation variations)
    - Fast motion (noise-like artifacts)
    """

    def __init__(self, seed: int = 42):
        """Initialize with optional seed."""
        self.rng = np.random.RandomState(seed)
        logger.info("RobustnessAugmentation ready")

    # ── Compression simulation ────────────────────────────────────────────────

    def simulate_jpeg_compression(
        self, image: np.ndarray, quality: int = 85
    ) -> np.ndarray:
        """Simulate JPEG compression artifacts.
        
        Args:
            image: RGB uint8 or float32 image.
            quality: JPEG quality (0-100), lower = more artifacts.
            
        Returns:
            Compressed image (same format as input).
        """
        is_float = image.dtype == np.float32
        if is_float:
            image_u8 = (image * 255).astype(np.uint8)
        else:
            image_u8 = image
        
        # Encode and decode with JPEG
        _, encoded = cv2.imencode(".jpg", image_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        
        if is_float:
            return decoded.astype(np.float32) / 255.0
        return decoded

    # ── Blur simulation ──────────────────────────────────────────────────────

    def simulate_motion_blur(
        self, image: np.ndarray, kernel_size: int = 7, angle: Optional[float] = None
    ) -> np.ndarray:
        """Simulate motion blur.
        
        Args:
            image: RGB image.
            kernel_size: Motion kernel size (odd number, 3-31).
            angle: Blur direction in degrees (None = random).
            
        Returns:
            Blurred image.
        """
        kernel_size = int(kernel_size) | 1  # Ensure odd
        kernel_size = max(3, min(31, kernel_size))
        
        if angle is None:
            angle = self.rng.uniform(0, 180)
        
        # Create motion blur kernel manually (line kernel rotated by angle)
        k = kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        # draw a horizontal line in the center
        cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1, thickness=1)

        # rotate kernel by angle around center
        center = (k // 2, k // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(kernel, rot_mat, (k, k), flags=cv2.INTER_LINEAR)

        # normalize
        if rotated.sum() != 0:
            rotated = rotated / rotated.sum()

        return cv2.filter2D(image, -1, rotated)

    def simulate_focus_blur(
        self, image: np.ndarray, sigma: float = 2.0
    ) -> np.ndarray:
        """Simulate focus blur (Gaussian).
        
        Args:
            image: RGB image.
            sigma: Blur standard deviation.
            
        Returns:
            Blurred image.
        """
        kernel_size = int(sigma * 6) | 1
        kernel_size = max(3, min(31, kernel_size))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # ── Brightness variation ──────────────────────────────────────────────────

    def adjust_brightness(
        self, image: np.ndarray, factor: float = 1.0
    ) -> np.ndarray:
        """Adjust image brightness.
        
        Args:
            image: RGB image (uint8 or float32).
            factor: Brightness multiplier (0.5-2.0 typical).
            
        Returns:
            Adjusted image (same format).
        """
        is_float = image.dtype == np.float32
        
        if is_float:
            adjusted = np.clip(image * factor, 0, 1)
        else:
            adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        return adjusted

    def adjust_contrast(
        self, image: np.ndarray, factor: float = 1.0, center: float = 0.5
    ) -> np.ndarray:
        """Adjust image contrast.
        
        Args:
            image: RGB image.
            factor: Contrast multiplier (0.5-2.0 typical).
            center: Center value for contrast adjustment (0.5 for mid-gray).
            
        Returns:
            Adjusted image.
        """
        is_float = image.dtype == np.float32
        
        if is_float:
            img = image
        else:
            img = image.astype(np.float32) / 255.0
        
        adjusted = center + (img - center) * factor
        adjusted = np.clip(adjusted, 0, 1)
        
        if is_float:
            return adjusted
        else:
            return (adjusted * 255).astype(np.uint8)

    # ── Noise injection ───────────────────────────────────────────────────────

    def add_gaussian_noise(
        self, image: np.ndarray, std: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise.
        
        Args:
            image: RGB image.
            std: Noise standard deviation (0.01-0.05 typical).
            
        Returns:
            Noisy image.
        """
        is_float = image.dtype == np.float32
        
        if is_float:
            img = image
        else:
            img = image.astype(np.float32) / 255.0
        
        noise = self.rng.normal(0, std, img.shape)
        noisy = np.clip(img + noise, 0, 1)
        
        if is_float:
            return noisy
        else:
            return (noisy * 255).astype(np.uint8)

    def add_salt_pepper_noise(
        self, image: np.ndarray, salt_pepper_ratio: float = 0.01
    ) -> np.ndarray:
        """Add salt-and-pepper (impulse) noise.
        
        Args:
            image: RGB image.
            salt_pepper_ratio: Fraction of pixels to corrupt (0.001-0.05).
            
        Returns:
            Noisy image.
        """
        output = image.copy()
        total_pixels = output.shape[0] * output.shape[1]
        num_pixels = int(total_pixels * salt_pepper_ratio)
        
        coords = [self.rng.randint(0, s, num_pixels) for s in image.shape[:2]]
        
        # Salt (white): half
        salt_count = num_pixels // 2
        output[coords[0][:salt_count], coords[1][:salt_count]] = 255 if image.dtype == np.uint8 else 1.0
        
        # Pepper (black): other half
        output[coords[0][salt_count:], coords[1][salt_count:]] = 0
        
        return output

    # ── Geometric transformations ─────────────────────────────────────────────

    def random_scale(
        self, image: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """Random scaling with padding/cropping.
        
        Args:
            image: RGB image.
            scale_range: (min_scale, max_scale).
            
        Returns:
            Scaled image (same size as input).
        """
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        h, w = image.shape[:2]
        
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        
        if scale > 1.0:
            # Crop from center
            dh, dw = new_h - h, new_w - w
            y1, x1 = dh // 2, dw // 2
            return scaled[y1:y1+h, x1:x1+w]
        else:
            # Pad with edge values
            dh, dw = h - new_h, w - new_w
            y1, x1 = dh // 2, dw // 2
            padded = np.pad(scaled, ((y1, dh-y1), (x1, dw-x1), (0, 0)), mode="edge")
            return padded[:h, :w]

    def random_rotate(
        self, image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)
    ) -> np.ndarray:
        """Random rotation.
        
        Args:
            image: RGB image.
            angle_range: (min_angle, max_angle) in degrees.
            
        Returns:
            Rotated image (same size, with padding).
        """
        angle = self.rng.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated

    # ── Test-time augmentation (TTA) ──────────────────────────────────────────

    def augment_inference(
        self, image: np.ndarray, num_augmentations: int = 4
    ) -> List[np.ndarray]:
        """Generate augmented versions of image for ensemble voting.
        
        Args:
            image: RGB image.
            num_augmentations: Number of augmented versions (1-8).
            
        Returns:
            List of augmented images (includes original at position 0).
        """
        augmentations = [image]  # Original
        
        if num_augmentations >= 2:
            # Compression variants
            augmentations.append(self.simulate_jpeg_compression(image, quality=75))
        
        if num_augmentations >= 3:
            # Blur
            augmentations.append(self.simulate_motion_blur(image, kernel_size=5))
        
        if num_augmentations >= 4:
            # Brightness variation
            augmentations.append(self.adjust_brightness(image, factor=0.85))
        
        if num_augmentations >= 5:
            # Scale
            augmentations.append(self.random_scale(image, (0.9, 1.1)))
        
        if num_augmentations >= 6:
            # Noise
            augmentations.append(self.add_gaussian_noise(image, std=0.02))
        
        if num_augmentations >= 7:
            # Rotate
            augmentations.append(self.random_rotate(image, (-10, 10)))
        
        if num_augmentations >= 8:
            # Contrast
            augmentations.append(self.adjust_contrast(image, factor=1.2))
        
        return augmentations[:num_augmentations]

    # ── Combined training augmentation ────────────────────────────────────────

    def augment_training(self, image: np.ndarray) -> np.ndarray:
        """Aggressive augmentation for training.
        
        Args:
            image: RGB image.
            
        Returns:
            Heavily augmented image.
        """
        # Random sequence of augmentations
        output = image.copy()
        
        # 70% chance of compression
        if self.rng.rand() > 0.3:
            quality = self.rng.randint(60, 95)
            output = self.simulate_jpeg_compression(output, quality=quality)
        
        # 40% chance of blur
        if self.rng.rand() > 0.6:
            if self.rng.rand() > 0.5:
                output = self.simulate_motion_blur(output, kernel_size=self.rng.randint(3, 9))
            else:
                output = self.simulate_focus_blur(output, sigma=self.rng.uniform(1, 3))
        
        # 50% chance of brightness/contrast
        if self.rng.rand() > 0.5:
            output = self.adjust_brightness(output, factor=self.rng.uniform(0.8, 1.3))
        
        if self.rng.rand() > 0.5:
            output = self.adjust_contrast(output, factor=self.rng.uniform(0.8, 1.4))
        
        # 30% chance of noise
        if self.rng.rand() > 0.7:
            if self.rng.rand() > 0.5:
                output = self.add_gaussian_noise(output, std=self.rng.uniform(0.005, 0.02))
            else:
                output = self.add_salt_pepper_noise(output, salt_pepper_ratio=self.rng.uniform(0.001, 0.01))
        
        # 40% chance of geometric transform
        if self.rng.rand() > 0.6:
            if self.rng.rand() > 0.5:
                output = self.random_scale(output, scale_range=(0.85, 1.15))
            else:
                output = self.random_rotate(output, angle_range=(-20, 20))
        
        return output
