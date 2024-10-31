# Week02a

> 2024/09/10 이론 수업  

## Preview

> 텐서플로우 소개

## TODO

- [ ] [텐서플로우 초보자용 설명서](https://www.tensorflow.org/tutorials?hl=ko) 확인해 보기  

## Contents

- ## 텐서플로우

    > 머신러닝 라이브러리  
    > 구글  

- ## 텐서플로우의 장점

    > 편리하다(테라스 라는 api에 연결가능 == 높은 수준의 연결성을 제공)  
    > 풍부한 예제  
    > gpu뿐만 아니라 tpu(텐서프로세스유닛)으로도 비약적인 학습 시간 단축  
    > 확장성(대규모 데이터 셋)  
    > 분산 학습 지원  

- ## 텐서플로우의 특징

    > 그래프 기반 계산(노드, 간선(엣지))  
    > 데이터의 흐름을 쉽게 볼수 있어서 플로우  
    > 텐서(tensor 데이터가 담긴 하나의 컨테이너/데이터의 기본 단위)를 사용하여 데이터를 표현  

- ## 텐서

    > 데이터의 기본 단위  

- ## 텐서의 랭크(차원의 개수)

    > 스칼라(0차원),백터(1차원),행렬(2차원),3차원 배열  

- ## 텐서 생성

    ```python
    import tensorflow as ts

    ts.zeros(shape)
    ts.ones(shape)
    ts.random.normal(shape)
    ts.constant(vale)
    ```
    > [예제01](./Example01.py), [예제02](./Example02.py), [예제03](./Example03.py)  

- ## 텐서 변환  

    ```python
    tf.cast()
    ```
    > tf.cast() 함수를 사용하면 텐서의 데이터 타입을 변경할 수 있습니다.  
    > [예제04](./Example04.py)  
    ```python
    tf.convert_to_tensor()
    ```
    > tf.convert_to_tensor()함수를 사용하면 다양한 데이터 소스(배열, 리스트 등)를 텐서의 형태로 변경할 수 있습니다.  
    > [예제05](./Example05.py)  
    ```python
    tf.reshape()
    ```
    > tf.reshape() 함수를 사용하면 텐서의 형태를 변경할 수 있습니다.  
    > 기존 numpy의 것과 같다  
    > [예제06](./Example06.py)

- ## 백터의 내적 계산

    > [예제07](Example07.py)  

- ## 텐서플로우 with: numpy, matplotlib

    > numpy : 고성능 배열연산을 지원하는 라이브러리  
    > matplotlib : 데이터를 시각화 하는 라이브러리

- ## 매개변수 

    > 계속해서 변화하는 값(가중치)  
    > 머신러닝 -> 매개변수를 최적화 하는 과정

- ## 텐서플로우를 이용한 데이터 스파이크 감지
  
    > [예제08](./Example08.py)  

- ## 텐서플로우의 체크포인트

    > [예제09](./Example09.py), [예제10](./Example10.py)  

---

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

- ## 텐서플로우를 활용한 선형 회귀 코드

    > [예제11](./Example11.py)
