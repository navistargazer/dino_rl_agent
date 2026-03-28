# Chrome Dino RL Agent

## 프로젝트 개요
본 프로젝트는 구글 크롬 브라우저의 오프라인 공룡 게임(Chrome Dinosaur Game)을 플레이하는 심층 강화학습(Deep Reinforcement Learning) 에이전트입니다. DQN(Deep Q-Network) 알고리즘과 CNN(Convolutional Neural Network) 구조를 채택하여, 화면의 픽셀 데이터를 직접 시각적 상태(State)로 입력받아 최적의 행동(Action: 점프, 숙이기, 대기)을 도출하도록 설계되었습니다.

## 주요 기능 및 특징
- **실시간 화면 인식 기반 상태 추출**: `mss`와 OpenCV 프레임워크를 결합해 복수 모니터 환경에서도 템플릿 매칭 기반으로 단일 게임 영역을 자동 탐색합니다. 인게임의 주/야간 테마 변경에 의한 영향을 제한하기 위해 Canny Edge Detection 기반의 전처리를 수행합니다.
- **Experience Replay 및 병렬 네트워크 구조**: 4-Frame Stacked Buffer를 구성하여 시퀀셜한 데이터를 학습합니다. Online Network와 Target Network를 분리해 타겟 값의 변동폭을 통제하고, Replay 메모리 버퍼 기법을 활용하여 데이터의 상관관계를 낮춤으로써 학습의 수렴 안정성을 확보했습니다.
- **H/W 가속 이식성**: 하드웨어 리소스를 자동으로 판별하여 CUDA 계열 GPU와 Apple Silicon(MPS)에 대한 연산 가속을 지원합니다.
- **성능 평가 및 로깅 툴 내장**: TensorBoard 로깅 시스템 모듈화를 통하여 생존 시간, 리워드 누적 합, 탐험률(Epsilon) 감소 지표 및 네트워크의 최고 Q값을 실시간으로 추적 및 시각화합니다.

## 디렉토리 구조 및 핵심 모듈
- `train_agent.py` : 강화학습 프로세스 메인 엔트리 스크립트. 에피소드 초기화 및 진행 루프 관리와 최적 가중치 갱신 작업을 담당.
- `test_agent.py` : 훈련 완료된 `.pth` 모델을 호출해 의사결정 정책 평가용 무탐험(Epsilon = 0) 플레이 스크립트.
- `dino_env.py` : 행동 수행, 보상 체계(Reward Function) 산정, Terminal State 판별 등 에이전트와 환경 상호작용 인터페이스 구현 래퍼.
- `dqn_cnn.py` : PyTorch를 통한 신경망 모델 아키텍쳐. Feature Extraction을 위한 합성곱 계층과 행동 기댓값 추론 전결합(FC) 계층 지원.
- `get_state.py` : 화면 픽셀 스크랩 및 OpenCV 연산. 논리 해상도와 물리 해상도 매핑 연산을 통해 프레임을 텐서 데이터로 전처리 수행.
- `actions.py` : `pyautogui` I/O 바인딩. 동시 입력 및 키업/다운 이벤트를 통해 상태머신 기반 조작 관리.
- `replay_buffer.py` / `train_buffer.py` : Transition 기록을 수집하고, 손실 역전파(Backpropagation) 최적화를 수행하는 오프폴리시(Off-policy) 학습 지원 모듈. 

## 환경 요구사항
Python 환경 하에 구성 가능하며, 다음과 같은 주요 패키지가 상호 호환되어야 합니다.

```bash
matplotlib
mss
numpy
pyautogui
torch
opencv-python
tensorboard
```

## 실행 가이드

### 1. 패키지 로드 및 초기 설정
저장소 클론 후 명령어를 통해 프로젝트 운영에 필요한 모든 서드파티 라이브러리를 동기화합니다.
```bash
pip install -r requirements.txt
```
웹 브라우저에서 `chrome://dino/` 주소를 입력하여 게임을 화면에 노출시킵니다. 시스템은 `template.png`를 기준으로 좌표를 교정하므로 화면을 가리지 않아야 합니다.

### 2. 학습 파이프라인 가동
아래 명령을 입력 시, 강화학습 루틴이 즉각 실행됩니다. 점진적으로 Epsilon 값이 붕괴하면서 모델을 학습함과 동시에, `models/` 디렉토리에 가장 오랜 생존 시간을 보인 Weight가 저장 및 갱신됩니다.
```bash
python train_agent.py
```
학습 추이(Loss, Q-Value, Survival Time 등) 모니터링은 별도의 백그라운드 프로세스에서 TensorBoard를 활성화하여 접속합니다.
```bash
tensorboard --logdir=runs
```

### 3. 모델 성능 검증
평가(Validation) 스크립트를 사용하여 훈련 완료된 아키텍처의 의사결정 전략을 실시간 게임에서 검증합니다.
```bash
python test_agent.py
```
