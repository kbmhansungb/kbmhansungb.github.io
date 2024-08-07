## DecisionTree
결정 트리는 조건부 제어문만을 포함하는 알고리즘을 나타내는 방법 중 하나입니다. 계층적 모델로 트리 구조의 형태를 가지고 있습니다. 이 모델은 의사 결정 분석 및 기계 학습에 널리 사용됩니다.

### Information Gain
의사결정트리 모델에서 정보이득(Information Gain)은 의사결정트리가 분류 결정을 내리기 위하여 사용하는 중요한 지표입니다. 정보이득은 특정 기준(특성)으로 데이터를 분할했을 때 얻을 수 있는 데이터 순도(purity) 또는 불순도(impurity)의 개선 정도를 측정합니다. 간단히 말해서, 정보이득은 분기 전후의 데이터 불순도 차이를 의미합니다.

분기를 통해 얻을 수 있는 정보이득이 최대가 되도록 선택하는 것이 의사결정트리 학습의 핵심입니다. 이는 모델이 더 정확한 예측을 할 수 있도록 도와주며, 결국 더 효과적으로 데이터를 분류하고 의사 결정을 내릴 수 있게 합니다.

정보이득이 높은 특성을 우선적으로 선택함으로써, 의사결정트리는 가능한 적은 수의 질문으로 데이터를 효과적으로 분리할 수 있습니다. 이로 인해 모델의 복잡성이 감소하고, 과적합(overfitting) 가능성이 줄어들며, 결과적으로 향상된 예측 성능을 기대할 수 있습니다.

따라서 정보이득은 의사결정트리 학습 과정에서 최적의 질문을 선택하고, 효율적이고 정확한 모델을 만드는 데 중요한 역할을 합니다.

정보이득(Information Gain, IG)을 측정하는 방법은 다음과 같은 수학식으로 표현됩니다:

$$
IG(D_p, f) = I(D_p) - \sum_{j=1}^{m} \left( \frac{N_j}{N_p} \right) I(D_j)
$$

여기서 각 기호는 다음과 같은 의미를 가집니다:

- $IG:$ 정보이득
- $D_p:$ 부모 노드의 데이터 집합
- $f:$ 분기를 결정하는 특징 (예: 꽃잎 길이, 꽃잎 너비 등)
- $m:$ 자식 노드의 개수. 이진 의사결정트리에서는 \(m = 2\)
- $N_p:$ 부모 노드의 데이터 수
- $N_j:$ \(j\)번째 자식 노드의 데이터 수
- $I(D_p):$ 부모 노드의 데이터 불순도(impurity)
- $I(D_j):$ \(j\)번째 자식 노드의 데이터 불순도

이 식을 통해, 특정 특징 $f$을 기준으로 데이터를 분할했을 때 분할 전과 분할 후의 불순도 변화량을 측정할 수 있습니다. 정보이득이 클수록 해당 특징을 기준으로 분할했을 때 데이터의 순도가 개선되는 것이며, 이는 모델의 분류 성능 개선에 기여하게 됩니다. 따라서 의사결정트리를 구성할 때는 정보이득이 최대가 되는 특징을 선택하여 데이터를 분기하게 됩니다.

## 참고
* [Wiki, "Dicision tree"](https://en.wikipedia.org/wiki/Decision_tree)
