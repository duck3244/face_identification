import os
import pytest
from validators import validate_image_path, validate_threshold, validate_top_k


class TestValidateImagePath:
    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_image_path("/nonexistent/image.jpg")

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            validate_image_path(123)

    def test_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")
        with pytest.raises(ValueError, match="지원하지 않는 이미지 형식"):
            validate_image_path(str(txt_file))

    def test_valid_image(self, tmp_image):
        # 유효한 이미지 경로는 예외 없이 통과
        validate_image_path(tmp_image)

    def test_supported_extensions(self, tmp_path):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            img_file = tmp_path / f"test{ext}"
            img_file.write_bytes(b'\x00')
            validate_image_path(str(img_file))


class TestValidateThreshold:
    def test_valid_values(self):
        validate_threshold(0.0)
        validate_threshold(0.5)
        validate_threshold(1.0)

    def test_below_range(self):
        with pytest.raises(ValueError):
            validate_threshold(-0.1)

    def test_above_range(self):
        with pytest.raises(ValueError):
            validate_threshold(1.1)

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            validate_threshold("0.5")

    def test_int_is_valid(self):
        validate_threshold(0)
        validate_threshold(1)


class TestValidateTopK:
    def test_valid_values(self):
        validate_top_k(1)
        validate_top_k(10)

    def test_zero(self):
        with pytest.raises(ValueError):
            validate_top_k(0)

    def test_negative(self):
        with pytest.raises(ValueError):
            validate_top_k(-1)

    def test_float_is_invalid(self):
        with pytest.raises(ValueError):
            validate_top_k(1.5)
