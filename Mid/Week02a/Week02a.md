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





