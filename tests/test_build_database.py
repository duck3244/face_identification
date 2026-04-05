import os
import pytest
from unittest.mock import patch, MagicMock

from build_database import build_database


class TestBuildDatabase:
    @patch('build_database.FaceRecognitionSystem')
    def test_build_database_no_dir(self, mock_system_cls, tmp_path, monkeypatch):
        """DATABASE_DIR이 없을 때 face_count 에러 없이 동작"""
        monkeypatch.setattr('build_database.DATABASE_DIR', str(tmp_path / "nonexistent"))
        monkeypatch.setattr('build_database.DATABASE_FILE', str(tmp_path / "db.pkl"))

        mock_system = MagicMock()
        mock_system_cls.return_value = mock_system

        result = build_database()
        # face_count가 0이므로 save_database 호출되지 않아야 함
        mock_system.save_database.assert_not_called()

    @patch('build_database.FaceRecognitionSystem')
    def test_build_database_empty_dir(self, mock_system_cls, tmp_path, monkeypatch):
        """빈 디렉토리일 때 정상 동작"""
        db_dir = tmp_path / "face_db"
        db_dir.mkdir()
        monkeypatch.setattr('build_database.DATABASE_DIR', str(db_dir))
        monkeypatch.setattr('build_database.DATABASE_FILE', str(tmp_path / "db.pkl"))

        mock_system = MagicMock()
        mock_system_cls.return_value = mock_system

        result = build_database()
        mock_system.save_database.assert_not_called()

    @patch('build_database.FaceRecognitionSystem')
    def test_build_database_with_images(self, mock_system_cls, tmp_path, monkeypatch):
        """이미지가 있는 디렉토리에서 정상 동작"""
        db_dir = tmp_path / "face_db"
        person_dir = db_dir / "John"
        person_dir.mkdir(parents=True)
        (person_dir / "face1.jpg").write_bytes(b'\x00')
        (person_dir / "face2.png").write_bytes(b'\x00')

        monkeypatch.setattr('build_database.DATABASE_DIR', str(db_dir))
        monkeypatch.setattr('build_database.DATABASE_FILE', str(tmp_path / "db.pkl"))

        mock_system = MagicMock()
        mock_system.add_face_to_database.return_value = True
        mock_system.database = MagicMock()
        mock_system_cls.return_value = mock_system

        result = build_database()
        assert mock_system.add_face_to_database.call_count == 2
        mock_system.save_database.assert_called_once()
