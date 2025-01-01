<div align="center">
  
# anime-face-cropper 

애니메이션 영상에서 개별 프레임을 추출한 뒤, YOLO 모델을 사용해 얼굴을 감지하고, 1:1 비율로 자른 이미지를 저장합니다.<br>
필요할 경우 검은색 패딩을 추가하여 얼굴 이미지를 정사각형으로 보정합니다.

</div>


## 📋 요구 사항

- 🐍 Python 3.11 권장(개발환경)
- 필요한 라이브러리 설치:
  ```bash
  pip install opencv-python numpy ultralytics tqdm
  ```
- `models/` 폴더에 [Fuyucch1/yolov8_animeface](https://github.com/Fuyucch1/yolov8_animeface)에서 release된 YOLO 모델 파일 (`yolov8x6_animeface.pt`)을 다운로드해 추가하세요.

---

## 🛠️ 사용 방법

1. 📂 `videos/` 폴더에 비디오 파일을 넣습니다.
2. 🏃 코드를 실행합니다
   ```bash
   python main.py
   ```
3. 코드를 실행 후 옵션을 선택합니다
   - 🖥️ 사용할 device
     - `CPU`
     - `GPU`
     - `MPS`
     
   - 🎬 처리 모드
     - `Process 1 video`
     - `Process 1 video from a specific frame`
     - `rocess all videos`

## 📤 출력

- 📸 **프레임**: `frames/<video_name>/frame_<number>.jpg`로 저장.
- 😎 **crop된 얼굴**: `crops/<video_name>/frame_<number>_face_<index>.jpg`로 저장.

