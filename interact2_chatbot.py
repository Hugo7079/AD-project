import streamlit as st
import asyncio
import logging
import time
import subprocess
import json
from transformers import pipeline
import warnings
import re
import speech_recognition as sr
import whisper
import soundfile as sf
import uuid
import ollama
import os
import torch
import pandas as pd
import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from sentence_transformers import SentenceTransformer, util
from opencc import OpenCC
from pydub import AudioSegment
from gtts import gTTS
import base64
from io import BytesIO
import string
import math

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_whisper_model(device):
    model = whisper.load_model("medium", device=device)
    return model

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_ckip_models():
    ws_driver = CkipWordSegmenter(model="bert-base")
    pos_driver = CkipPosTagger(model="bert-base")
    return ws_driver, pos_driver

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_sentence_transformer():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

# ==================== GPU或CPU =====================
use_cuda = torch.cuda.is_available()
if use_cuda:
    try:
        torch.cuda.set_per_process_memory_fraction(0.8)
        device = torch.device("cuda")
        print("使用GPU")
    except Exception as e:
        print(f"CUDA 初始化失敗，使用 CPU：{e}")
        device = torch.device("cpu")
else:
    print("未檢測到 GPU，使用 CPU")
    device = torch.device("cpu")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 幫助函式：不重疊播放 =====================
def estimate_audio_duration(text, speed=1.5):
    """
    粗估朗讀秒數: 假設 1秒 朗讀 5字
    """
    base_speed = 5.0
    length = len(text)
    base_time = length / base_speed
    return base_time + 1

def text_to_speech_and_wait(text, speed=1.5):
    """
    文字轉語音並播放，避免與下一段重疊
    """
    try:
        tts = gTTS(text=text, lang='zh')
        with BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        autoplay = "autoplay"

        audio_html = f'''
        <audio id="audioPlayer" {autoplay}>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.getElementById("audioPlayer");
            audio.play().then(() => {{
                audio.playbackRate = {speed};
            }}).catch(error => {{
                console.error("音頻播放失敗:", error);
            }});
        </script>
        '''
        st.components.v1.html(audio_html, height=0)

        dur = estimate_audio_duration(text, speed) + 1.0
        time.sleep(dur)

    except Exception as e:
        st.error(f"文字轉語音時發生錯誤：{e}")
        time.sleep(2)

# ==================== CognitiveTestHelper =====================
class CognitiveTestHelper:
    def __init__(self, model):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="uer/roberta-base-finetuned-jd-binary-chinese",
            device=-1
        )
        self.model = model

    async def perform_rag(self, user_input, selected_model_name, current_question, previous_answers="", temperature_level=0.8, max_tokens=300):
        """
        與 Ollama 模型溝通
        """
        try:
            assistant_instructions = f"""
                您是一名專業的認知功能測驗助手，負責與使用者進行對話，收集他們的回答並分析他們的表現，評估失智症風險。請根據以下指引進行操作：

                1. 測驗步驟：
                - 此階段為「自主性語言表達測驗」，共 6 個問題。
                - 每次只展示一個問題，鼓勵回答超過1分鐘或100字。
                - 若回答不足或不相干，請溫和地引導補充。

                2. 先前回答：
                {previous_answers}

                3. 當前問題：
                {current_question}

                4. 用戶回答：
                {user_input}

                5. 指導原則：
                - 全部用繁體中文。
                - 客觀、溫和但堅定、適度引導。
            """
            response = await asyncio.to_thread(
                ollama.chat,
                model=selected_model_name,
                messages=[{"role": "user", "content": assistant_instructions}]
            )
            response_content = response['message']['content']
            truncated_response = response_content[:max_tokens]
            return truncated_response
        except Exception as e:
            logging.error(f"生成回應時發生錯誤：{e}")
            st.error(f"無法生成回應：{e}")
            return "對不起，出現錯誤，請稍後再試。"

    def is_relevant(self, question, answer, threshold=0.2):
        """
        簡單相干性檢測
        """
        try:
            score = util.pytorch_cos_sim(
                self.model.encode(question, convert_to_tensor=True),
                self.model.encode(answer, convert_to_tensor=True)
            )
            return score.item() >= threshold
        except Exception as e:
            logging.error(f"相干性檢測出現錯誤: {e}")
            return True

# ==================== CognitiveTestApp =====================
class CognitiveTestApp:
    def __init__(self):
        self.helper = CognitiveTestHelper(load_sentence_transformer())
        self.initialize_session_state()
        self.questions = [
            "請問您的家鄉在哪裡？住了多久？有沒有特殊的回憶？",
            "颱風來了通常會發生什麼事？通常會做什麼準備？",
            "您昨天晚餐吃了什麼？可以告訴我細節嗎？",
            "假如您要泡一壺茶或是一杯咖啡，有什麼步驟呢？",
            "您最近喜歡看什麼節目？電視劇、廣播、網路節目都可以。",
            "一年四季當中，您最喜歡哪個季節？有什麼特殊的回憶嗎？"
        ]
        self.whisper_model = load_whisper_model(device)
        self.ckip_ws_driver, self.ckip_pos_driver = load_ckip_models()
        self.opencc = OpenCC('s2t')

    def initialize_session_state(self):
        defaults = {
            'step': 0,
            'conversation_history': [],
            'audio_data': [],
            'analysis_done': False,
            'intro_audio_played': False,
            'selected_model_name': "gemma2",
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def record_audio(self, duration=30):
        recognizer = sr.Recognizer()
        info_placeholder = st.empty()
        info_placeholder.info(f"錄音中，請在 {duration} 秒內作答...")
        try:
            with sr.Microphone() as source:
                audio = recognizer.record(source, duration=duration)
            text_to_speech_and_wait("錄音結束，正在轉錄音檔...")
            info_placeholder.info("轉錄中，請稍後...")
            time.sleep(5)
            info_placeholder.empty()
        except sr.WaitTimeoutError:
            info_placeholder.error("錄音超時。")
            time.sleep(2)
            info_placeholder.empty()
            return None

        save_dir = os.path.join(os.getcwd(), "audio_files")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        unique_filename = f"recorded_audio_{uuid.uuid4()}.wav"
        save_path = os.path.join(save_dir, unique_filename)

        with open(save_path, "wb") as f:
            f.write(audio.get_wav_data())

        return save_path

    def transcribe_with_whisper(self, audio_file_path):
        result = self.whisper_model.transcribe(audio_file_path, language="zh")
        return result["text"]

    def auto_ask_and_record_question(self, question, step_number):
        st.write(f"### 第 {step_number} 題")
        st.write(question)

        if step_number == 1 and not st.session_state.intro_audio_played:
            intro_text = (
                "您好，歡迎參加自主性語言表達測驗，總共有六個問題。"
                "回答時盡量詳細描述，每題錄音30秒，若回答不足可能會請您補充。"
                "準備好後開始第一題。"
            )
            text_to_speech_and_wait(intro_text)
            st.session_state.intro_audio_played = True

        text_to_speech_and_wait(f"現在是第 {step_number} 題")
        text_to_speech_and_wait(question)

        max_attempts = 2
        for attempt in range(1, max_attempts+1):
            # 倒數提示
            countdown_area = st.empty()
            for i in range(3,0,-1):
                countdown_area.info(f"{i} 秒後開始錄音...")
                time.sleep(1)
            countdown_area.empty()

            audio_file = self.record_audio(duration=30)
            if audio_file is None:
                break

            st.session_state.audio_data.append(audio_file)
            try:
                transcribed_text = self.transcribe_with_whisper(audio_file)
            except Exception as e:
                st.error(f"轉錄失敗：{e}")
                break

            # 檢驗回答相干性及字數
            relevance = self.helper.is_relevant(question, transcribed_text, threshold=0.2)
            word_count = len(transcribed_text)
            st.session_state.conversation_history.append({
                "question": question,
                "answer": transcribed_text,
                "relevant": relevance
            })

            # 若回答相干且字數夠，就下一題；否則請模型給建議
            if word_count >= 100 and relevance:
                if step_number == len(self.questions):
                    text_to_speech_and_wait("測驗完成，接下來將進行結果分析...")
                else:
                    text_to_speech_and_wait("好的，您的回答已足夠。讓我們繼續下一題。")
                break
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                prev_ans = [
                    f"問題：{c['question']} 回答：{c['answer']}" 
                    for c in st.session_state.conversation_history[:-1] 
                    if c.get('relevant', True)
                ]
                prev_string = "\n".join(prev_ans)
                response = loop.run_until_complete(
                    self.helper.perform_rag(
                        user_input=transcribed_text,
                        selected_model_name=st.session_state['selected_model_name'],
                        current_question=question,
                        previous_answers=prev_string
                    )
                )
                loop.close()

                # 如果測驗尚未完成，才唸出模型回覆
                if step_number < len(self.questions):
                    text_to_speech_and_wait(response)

                if attempt == max_attempts:
                    # 若還是不足，直接跳下一題
                    if step_number == len(self.questions):
                        text_to_speech_and_wait("測驗完成，請稍候進行結果分析。")
                    else:
                        text_to_speech_and_wait("尚未達到建議字數，但先進入下一題。")

    def perform_analysis(self):
        if st.session_state.get('analysis_done', False):
            return

        combined = AudioSegment.empty()
        for audio_file_path in st.session_state.audio_data:
            try:
                sound = AudioSegment.from_file(audio_file_path)
                combined += sound
            except Exception as e:
                st.error(f"合併音頻檔案時發生錯誤：{e}")
                return

        combined_audio_path = os.path.join(os.getcwd(), "audio_files", "combined_audio.wav")
        try:
            combined.export(combined_audio_path, format='wav')
        except Exception as e:
            st.error(f"導出合併音頻時發生錯誤：{e}")
            return

        st.session_state.analysis_done = True
        text_to_speech_and_wait("以上是分析結果，感謝您的參與。")
        st.info("測驗已完成，如需重來請重新執行。")
        os._exit(0)

    def main(self):
        st.title("自主性語言表達測驗")

        if st.session_state.step == 0:
            st.write("歡迎參加測驗，請按下開始。")
            if st.button("開始測驗"):
                st.session_state.step = 1
                st.experimental_rerun()

        elif 1 <= st.session_state.step <= len(self.questions):
            question = self.questions[st.session_state.step - 1]
            self.auto_ask_and_record_question(question, st.session_state.step)
            st.session_state.step += 1
            st.experimental_rerun()

        elif st.session_state.step == len(self.questions) + 1:
            st.success("問題已結束，開始進行分析...")
            time.sleep(2)
            self.perform_analysis()

# ==================== 主程式 =====================
if __name__ == "__main__":
    app = CognitiveTestApp()
    app.main()
