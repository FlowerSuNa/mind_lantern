
from typing import List, Dict, Any
from pydrive.drive import GoogleDrive

def upload_to_drive(drive: GoogleDrive, google_drive_id: str, filename: str, data: Any) -> None:
    """
    데이터 구글 드라이브에 저장

    Args:
        drive (GoogleDrive): 구글 드라이브 객체
        google_drive_id (str): 구글 드라이브 폴더 ID
        filename (str): 업로드할 파일 이름
        data (any): 업로드할 데이터
    """
    file = drive.CreateFile({
        'title': filename,
        'parents': [{'id': google_drive_id}]
    })
    file.SetContentString(data)
    file.Upload()
    return None