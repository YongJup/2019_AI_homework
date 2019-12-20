# 휴대폰의 스펙으로 적절한 판매가격대를 선정하자

## 과제요약

- 휴대폰의 스펙(CPU clock, RAM 등)과  판매가격과의 관계를 찾으려한다.
- 실제 가격을 예측하지는 않고 가격이 얼마나 높은지의 가격의 범위를 찾으려 한다.

---

## 사용 데이터소스

### 데이터의 출처

https://www.kaggle.com/iabhishekofficial/mobile-price-classification

### 데이터의 구조

| 컬럼 이름     | 컬럼 내용                                                    |
| ------------- | ------------------------------------------------------------ |
| battery_power | 배터리 용량(mAh)                                             |
| blue          | 블루투스 사용가능 (0 or 1)                                   |
| clock_speed   | CPU의 클럭속도                                               |
| dual_sim      | 듀얼심 지원여부 (0 or 1)                                     |
| fc            | 전면카메라의 화소 (mega pixles)                              |
| four_g        | 4G지원여부 (0 or 1)                                          |
| int_memory    | 내장메모리 용량 (기가바이트)                                 |
| m_dep         | 핸드폰 두께                                                  |
| mobile_wt     | 핸드폰 무게                                                  |
| n_cores       | CPU 코어개수                                                 |
| pc            | 후방카메라의 화소 (mega pixels)                              |
| px_height     | 액정 픽셀 높이                                               |
| px_width      | 액정 픽셀 가로                                               |
| ram           | RAM (기가바이트)                                             |
| sc_h          | 액정 세로길이 (cm)                                           |
| sc_w          | 액정 가로길이 (cm)                                           |
| talk_time     | 전화사용시 배터리 시간                                       |
| three_g       | 3G 사용가능 (0 or 1)                                         |
| touch_screen  | 터치스크린 존재여부 (0 or 1)                                 |
| wifi          | 와이파이 존재여부 (0 or 1)                                   |
| price_range   | 가격대; 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost). |

### 데이터 개수, 형식

2000개 * 21컬럼 / CSV



## 예측 종류

가격이 아닌 가격대를 예측하기때문에 분류를 해야 한다.



## 수업시간에 만들 라이브러리 불러오기

```python
from classification_util import ClassificationUtil as cu
```

> 라이브러리 내 클래스 ClassificationUtil을 이름이 길기 때문에 cu로 불러온다



## 데이터(CSV)가져오기/보이기

```python
gildong = cu() #cu로 길동 객체 생성
gildong.read('train.csv')
gildong.show()
```

> 실행 코드

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
battery_power    2000 non-null int64
blue             2000 non-null int64
clock_speed      2000 non-null float64
dual_sim         2000 non-null int64
fc               2000 non-null int64
four_g           2000 non-null int64
int_memory       2000 non-null int64
m_dep            2000 non-null float64
mobile_wt        2000 non-null int64
n_cores          2000 non-null int64
pc               2000 non-null int64
px_height        2000 non-null int64
px_width         2000 non-null int64
ram              2000 non-null int64
sc_h             2000 non-null int64
sc_w             2000 non-null int64
talk_time        2000 non-null int64
three_g          2000 non-null int64
touch_screen     2000 non-null int64
wifi             2000 non-null int64
price_range      2000 non-null int64
dtypes: float64(2), int64(19)
memory usage: 328.2 KB
None
   battery_power  blue  clock_speed  ...  touch_screen  wifi  price_range
0            842     0          2.2  ...             0     1            1
1           1021     1          0.5  ...             1     0            2
2            563     1          0.5  ...             1     0            2
3            615     1          2.5  ...             0     0            2
4           1821     1          1.2  ...             1     0            1

[5 rows x 21 columns]
(2000, 21)
```

> 출력 결과

- 입력 데이터 전체 21개 컬럼, 2000개의 데이터 모두를 불러왔다.
- 모든 데이터에 비어있는 데이터는 없다



## 데이터 분석

### 히트맵 그려보기

```python
gildong.heatmap()
```

![myplot](/home/yonajae/Documents/JNUCE/3_2/exam/myplot.png)

히트맵을 봤을때 가격과 관계가 높은것들을 바라보면

![myplot_edit](/home/yonajae/Documents/JNUCE/3_2/exam/myplot_edit.png)

- 배터리용량, 액정의 해상도, 램 과의 상관관계가 높게나온다

그러나 지금 갖고있는 데이터를 그대로 사용하기에는 값들의 단위와 크기 등이 모두 다르기 때문에 **데이터 전처리가 필요하다**

하지만 데이터를 전처리를 하는 방법이나 내용은 들은바가 없기 때문에 만들어둔 라이브러리로 한번 실행시켜보자.



## 예측하기

### 모든 컬럼 넣어보기

- 데이터 전처리를 컬럼을 제외하는것 이외에는 해보지 않았기때문에 일단 모든 컬럼을 이용해 분석해보기로 한다

```python
c = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 
     'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
gildong.ignore_warning()
gildong.run_svm(c, 'price_range')
gildong.run_neighbor_classifier(c, 'price_range', 10)
gildong.run_logistic_regression(c, 'price_range')
gildong.run_decision_tree_classifier(c, 'price_range')
```

> 실행코드

```
인식률: 95.83333333333334
인식률: 93.0
인식률: 64.5
인식률: 79.66666666666666
```

> 실행 결과



### 히트맵에서 관계가 높게 나왔던거 넣어보기

- 배터리용량, 내장메모리용량, 화면해상도, 램용량을 이용해서 예측을 해봤다

```py
c = ['battery_power', 'int_memory', 'px_height', 'px_width', 'ram']
gildong.run_svm(c, 'price_range')
gildong.run_neighbor_classifier(c, 'price_range', 10)
gildong.run_logistic_regression(c, 'price_range')
gildong.run_decision_tree_classifier(c, 'price_range')
```

> 실행코드

```
인식률: 94.0
인식률: 93.5
인식률: 54.50000000000001
인식률: 88.16666666666667
```

> 실행결과



## 결과

21개의 컬럼중 히트맵에서 상관계수가 크게 나왔던 5가지 컬럼만 이용해 학습시켰더니

결과는 크게 다르지 않고 논리회귀를 이용한 인식률이 줄어들고, 의사결정트리에서 인식률이 올라갔다.

정확한 이유는 알지 못하겠다

데이터 전처리쪽을 공부를 해보고, 여러 학습 알고리즘들을 알아보면 인식률을 올리는데 도움이 될 것같다.