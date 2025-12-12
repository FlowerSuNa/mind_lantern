import os
import re
import json

from typing import List, Dict, Any

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import yt_dlp
from yt_dlp.utils import DownloadError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import SUBTITLE_SUMMARY_SYSTEM_TEMPLATE, SUBTITLE_SUMMARY_HUMAN_TEMPLATE
from utils import upload_to_drive

import time
from google.api_core.exceptions import ResourceExhausted

import requests
import warnings
import logging

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')
logging.getLogger('yt_dlp').setLevel(logging.ERROR)

# Google Drive 인증 (브라우저 인증)
GAUTH = GoogleAuth()
GAUTH.LocalWebserverAuth()
DRIVE = GoogleDrive(GAUTH)

# LLM 정의
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.1
)


def summarize_subtitle(content: str) -> str:
    """
    주어진 자막(content)을 요약하는 함수

    Args:
        content (str): 유튜브 영상에서 추출된 자막 데이터

    Returns:
        str: 요약된 자막 문장(대화체 유지 및 필요 요약)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUBTITLE_SUMMARY_SYSTEM_TEMPLATE),
        ("human", SUBTITLE_SUMMARY_HUMAN_TEMPLATE)
    ])
    chain = prompt | LLM | StrOutputParser()
    return chain.invoke({"content": content})

if __name__ == "__main__":
    import sys

    playlist_id = sys.argv[1] # PLGiaCgd9PatfXH13hpDWkZrxDLq-vvGtj

    google_dirve_rt_id = os.getenv("GOOGLE_DRIVE_RT_ID")
    google_dirve_vtt_id = os.getenv("GOOGLE_DRIVE_VTT_ID")
    google_dirve_txt_id = os.getenv("GOOGLE_DRIVE_TXT_ID")

    # 자막 데이터 파일 탐색
    google_dirve_txt_id = os.getenv("GOOGLE_DRIVE_TXT_ID")
    file = DRIVE.ListFile({
        'q': f"'{google_dirve_txt_id}' in parents and title = '{playlist_id}.json' and trashed = false"
    }).GetList()

    # 자막 데이터 로드
    if file:
        original_contents = json.loads(file[0].GetContentString())

    print(f"📂 자막 데이터 로드 완료 : {len(original_contents)}개")

    # 초기화 
    local_json_path = f"{playlist_id}.json"
    if os.path.exists(local_json_path):
        # 중간 저장된 데이터 로드
        with open(local_json_path, "r", encoding="utf-8") as f:
            contents = json.load(f)
        print(f"📂 가공 데이터 로드 완료 : {len(contents)}개")
    else:
        contents = {}

    # 가공 안된 데이터 확인
    for i, key in enumerate(original_contents.keys()):
        if not contents.get(key):
            print(i, key)

    # 데이터 저장

    upload_to_drive(
        f'{playlist_id}.json', 
        json.dumps(contents, ensure_ascii=False, indent=2), 
        google_dirve_txt_id
    )
