import os
import re
import json

from typing import List, Dict, Any

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import yt_dlp
from yt_dlp.utils import DownloadError

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import upload_to_drive

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

def get_playlist_entries(playlist_url: str) -> List[Dict]:
    """ 
    유튜브 플레이리스트 URL 데이터 반환

    Args:
        playlist_url (str): 유튜브 플레이리스트 URL

    Returns:
        List[Dict]: 각 동영상에 대한 정보가 담긴 리스트
    """
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        return info['entries']
    
def get_subtitle_text(info_dict: Dict) -> str:
    """ 
    유튜브 자막 데이터 반환

    Args:
        info_dict (Dict): 유튜브 동영상 정보

    Returns:
        str: 자막 데이터
    """
    # 자막 가져오기
    subtitles = info_dict.get('subtitles') or info_dict.get('automatic_captions')
    
    if not subtitles or 'ko' not in subtitles:
        return None
    
    # 자막 가져오기
    subtitle_url = subtitles['ko'][0]['url']
    response = requests.get(subtitle_url)
    response.encoding = 'utf-8'
    return response.text

def clean_subtitle_text(title:str, content:Dict[str, Any]) -> str:
    """ 
    자막 데이터 1차 가공

    Args:
        title (str): 영상 제목
        content (dict): 자막 데이터

    Returns:
        str: 가공된 자막 데이터
    """
    segs = []
    for event in content['events']:
        if "segs" not in event:
            continue

        segs += event["segs"]
    
    cleaned_content = ' '.join([seg["utf8"] for seg in segs])
    cleaned_content = cleaned_content.replace('[박수]', ' ')
    cleaned_content = cleaned_content.replace('[웃음]', ' ')
    cleaned_content = cleaned_content.replace('[음악]', ' ')
    cleaned_content = cleaned_content.replace('(청중 웃음)', ' ')
    cleaned_content = cleaned_content.replace('(청중 박수)', ' ')
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()

    print(title)
    print('---'*20)
    print(f'{cleaned_content[:10]}...[{len(cleaned_content)}]', )
    print('==='*20)
    return cleaned_content

if __name__ == "__main__":
    import sys

    playlist_id = sys.argv[1] # PLGiaCgd9PatfXH13hpDWkZrxDLq-vvGtj

    # 유튜브 플레이리스트 URL 정보 로드
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    entries = get_playlist_entries(playlist_url)

    # 유튜브 자막 추출 생성자 정의
    ydl_opts = {
        'writesubtitles': True,           # 자막 다운로드 활성화
        'skip_download': True,            # 영상 자체는 다운로드하지 않음
        'subtitleslangs': ['ko'],         # 한국어 자막만 대상
        'writeautomaticsub': True,        # 자동 생성 자막(YouTube 자동 자막)도 허용
        'quiet': True,                    # 출력 로그 최소화
        'outtmpl': '-',                   # 파일 저장하지 않음 (stdout 출력 용도)
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)

    # 유튜브 영상에서 자막(트랜스크립트) 추출 체인 정의
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )
    transcript_chain = ChatPromptTemplate.from_messages([
        ("system", "당신은 유튜브 자막 생성 전문가 입니다."),
        ("human", "다음 유튜브 영상의 자막을 시간정보 없이 줄바꿈하며 한글로 정리해주세요: {video_url}")
    ]) | llm | StrOutputParser()

    # 유튜브 자막 추출 및 수집
    contents = {}
    for entry in entries:
        video_id = entry['id']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        transcript = "\n".join([item["text"] for item in raw])
        print(f'{video_id} [{len(transcript)}]')
        print('---'*20)
        print(transcript[:10])

        try:
            info_dict = ydl.extract_info(video_url, download=False)
        except DownloadError  as e:
            print(f"❌ 다운로드 에러 : {video_url} | 사유: {e}")
            continue

        title = info_dict.get('title')

        if "몰아보기" in title:
            print(f"❌ 몰아보기 영상 스킵 : {video_url}")
            continue
        elif not title:
            print(f"❌ 영상 정보 없음 : {video_url}")
            continue

        # 자막 추출
        try:
            text = get_subtitle_text(info_dict)
            if isinstance(text, dict) and "events" in text:
                subtitle = clean_subtitle_text(title, text)

            else:
                # 자막 추출 실패 시 LLM으로 자막 추출
                print(f"💡 LLM으로 자막 추출 : {video_url}")
                subtitle = transcript_chain.invoke({"video_url": video_url})

            contents[video_id] = {
                'title': title,
                'tags': info_dict.get('tags'),
                'video_url': video_url,
                'view_count': info_dict.get('view_count'),
                'duration': info_dict.get('duration'),
                'like_count': info_dict.get('like_count'),
                'channel': info_dict.get('channel'),
                'upload_date': info_dict.get('upload_date'),
                'subtitles_ko': subtitle
            }
        except Exception as e:
            print(f"❌ 자막 파싱 에러 : {video_url} | 사유: {e}")
            continue

    # 데이터 저장
    if contents:
        upload_to_drive(
            DRIVE,
            os.getenv("GOOGLE_DRIVE_TXT_ID"),
            f'{playlist_id}.json', 
            json.dumps(contents, ensure_ascii=False, indent=2)
        )
    else:
        print(f"❗ 데이터 없음 : {playlist_id}")
