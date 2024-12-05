# Exam

## Preview

> ?

## TODO

- [ ] ?

## Contents

- ## 텐서플로우의 특징

    > 그래프 기반 계산(노드, 간선(엣지))  
    > 데이터의 흐름을 쉽게 볼수 있어서 플로우  
    > 텐서(tensor 데이터가 담긴 하나의 컨테이너/데이터의 기본 단위)를 사용하여 데이터를 표현  

- ## 텐서

    > 데이터의 기본 단위  

- ## 텐서의 랭크(차원의 개수)

    > 스칼라(0차원),백터(1차원),행렬(2차원),3차원 배열  

- ## 회귀 분석 

    > 선형 회귀  
    > 다항식 회귀  
    > 로지스틱 회귀(0 or 1이진 분류)  

- ## 회귀 분석의 입력, 출력

    > 입력은 연속(Continuous),불연속(Discrete)둘다 가능하지만, 출력은 항상 연속(Continuous) 하다  

- ## 과적합과 과소적합

    > 과소적합(Underfit, 모델이 데이터를 충분히 반영하지 못함)  
    > 적절한 적합(Good fit, 적절하게 추이를 따라가게/일반화 능력이 우수하다)  
    > 과적합(Overfit, 로스는 제로지만 예측이 불가능하다)  

- ## 라소(LASSO)  

    > L1 단에서 특정 가중치는 0으로 만듦(무의미 하게 만듦)  

- ## 릿지(Ridge)  

    > L2 단에서 특정 가중치를 0에 가까워 지게 만듦(성격을 줄임)

- ## 선형 회귀?  

    > 활성 함수 : ReLU  
    > 비용 함수 : MSE  

- ## 1D 로지스틱 회기  

    > 활성 함수 : Sigmoid -> 바이어스를 넣지 않는다(y절편이 어떻고는 존제하지 않기 때문에)  
    > 비용 함수 : BCE(Binary crossentropy)  

- ## 로지스틱 회기?  

    > 활성 함수 : Sigmoid  
    > 비용 함수 : mean squared error  
    > 비용 함수 : BCE(Binary crossentropy)  

- ## Batch

    > 학습할때 학습 묶음 단위  
    > batch가 크면 -> 업데이트가 너무 느림.  
    > batch가 작으면 -> 업데이트가 너무 빨라서 최적화 하기에 좋지 않다.  

- ## 하이퍼 파라미터  

  - 머신 러닝에서 하이퍼 파라미터  
    > 머신 러닝이란 최적의 하이퍼 파라메타(애포크,학습률,등..)를 찾아가는 과정  

  - 종류  
    > 학습률  
    > 에포크  
    > 베치 사이즈  

--

- ## AI 가 정리해주는 머신러닝 기초  

    **하이퍼파라미터**는 모델이 학습할 때 **미리 설정해야 하는 값**들로, 모델의 성능에 큰 영향을 미칩니다. 이 용어들은 직접 모델 학습에 참여하지 않지만, 학습 과정을 조정하는 중요한 역할을 합니다. 아래에서 대표적인 하이퍼파라미터들을 정리해드리겠습니다.

    ### 1. **배치 크기 (Batch Size)**
    - **정의**: 한 번의 학습에서 모델이 학습할 데이터 샘플의 수입니다. 예를 들어, 배치 크기가 64이면, 모델은 한 번에 64개의 데이터를 처리하고 그 결과를 기반으로 가중치를 업데이트합니다.
    - **영향**:
        - 큰 배치 크기: 메모리 요구량이 크지만, 학습 과정이 안정적일 수 있습니다.
        - 작은 배치 크기: 더 자주 가중치가 업데이트되어 학습 속도가 빠르지만, 불안정할 수 있습니다.
    - **예**: 위 코드에서는 `batch_size=64`로 설정되었습니다.

    ### 2. **학습률 (Learning Rate)**
    - **정의**: 모델이 가중치를 업데이트할 때 변화시키는 크기를 결정하는 값입니다.
    - **영향**:
        - 학습률이 너무 크면 학습이 불안정하고 최적의 가중치에 도달하지 못할 수 있습니다.
        - 학습률이 너무 작으면 학습 속도가 느려집니다.
    - **예**: 코드에서 `learning_rate=0.1`로 설정되어 있습니다.

    ### 3. **에포크 (Epoch)**
    - **정의**: 학습 데이터 전체를 한 번 모두 학습하는 과정입니다. 1 에포크는 주어진 데이터셋을 한 번 다 학습했다는 의미입니다.
    - **영향**:
        - 너무 적은 에포크 수는 과소적합(underfitting)의 위험이 있습니다.
        - 너무 많은 에포크 수는 과적합(overfitting)의 위험이 있습니다.
    - **예**: 코드에서 `epochs=1000`으로 설정되어 있으며, 1000번 학습을 진행합니다.

    ### 4. **최적화 함수 (Optimizer)**
    - **정의**: 모델의 가중치를 어떻게 업데이트할지를 결정하는 알고리즘입니다.
    - **대표적인 최적화 함수**:
        - **SGD**: 확률적 경사 하강법 (Stochastic Gradient Descent)
        - **Adam**: 모멘텀과 학습률을 자동으로 조정하는 방식
        - **RMSprop**: 학습 속도를 높이고 진동을 줄이기 위한 알고리즘
    - **예**: 이 코드에서는 `SGD(learning_rate=0.1)`을 사용하여 경사 하강법을 통해 가중치를 조정합니다.

    ### 5. **손실 함수 (Loss Function)**
    - **정의**: 모델의 예측값과 실제값 간의 차이를 측정하는 함수입니다. 손실 값이 작을수록 모델이 데이터를 더 잘 예측하고 있다는 의미입니다.
    - **대표적인 손실 함수**:
        - **MSE**: 회귀 문제에서 주로 사용하는 `Mean Squared Error`(평균 제곱 오차)
        - **Cross-Entropy**: 분류 문제에서 많이 사용됩니다.
    - **예**: 코드에서 `loss='mse'`로 설정되어 있으며, 회귀 문제에 적합한 평균 제곱 오차를 사용합니다.

    ### 6. **활성화 함수 (Activation Function)**
    - **정의**: 신경망의 각 뉴런이 출력값을 결정할 때 사용하는 함수입니다.
    - **대표적인 활성화 함수**:
        - **ReLU**: 음수는 0으로 만들고, 양수는 그대로 출력. 심층 신경망에서 주로 사용.
        - **Sigmoid**: 0과 1 사이의 값으로 출력. 이진 분류에서 자주 사용.
        - **Softmax**: 다중 클래스 분류 문제에서 사용.
    - **예**: 코드에서는 `Dense` 레이어에서 `activation='relu'`로 설정되어 있습니다.

    ### 정리
    하이퍼파라미터는 **모델의 성능**과 **학습 과정**을 조정하는 중요한 요소들입니다. 적절한 값으로 설정해야 **과적합/과소적합**을 피하고, 모델이 효율적으로 학습할 수 있습니다.

    - **배치 크기**: 데이터를 나누는 단위
    - **학습률**: 가중치 업데이트 크기
    - **에포크**: 데이터 전체 학습 횟수
    - **최적화 함수**: 가중치 조정 방법
    - **손실 함수**: 예측과 실제 차이 측정
    - **활성화 함수**: 뉴런의 출력 결정