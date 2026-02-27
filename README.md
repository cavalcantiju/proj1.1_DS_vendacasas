Aqui está uma versão mais natural e com menos “cara de GPT”, mais no estilo de projeto acadêmico/portfólio:

---

# Regressão Linear – Previsão de Preço de Imóveis

## Objetivo

Este projeto aplica Regressão Linear (OLS) para analisar a relação entre o preço de venda de imóveis (`preco_de_venda`) e a variável `area_primeiro_andar`.

## Preparação dos dados

Primeiro, a variável resposta foi separada das variáveis explicativas:

```python
y = dados['preco_de_venda']
x = dados.drop(columns='preco_de_venda')
```

Em seguida, os dados foram divididos em treino e teste:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=230
)
```

Como o `statsmodels` com fórmula exige a variável dependente no mesmo DataFrame, o conjunto de treino foi reconstruído:

```python
df_train = X_train.copy()
df_train['preco_de_venda'] = y_train
```

## Modelo

O modelo foi estimado utilizando OLS:

```python
from statsmodels.formula.api import ols

modelo_0 = ols(
    'preco_de_venda ~ area_primeiro_andar',
    data=df_train
).fit()
```

A equação estimada foi:

preco_de_venda = 152900 + 6793.64 * area_primeiro_andar

O intercepto (152900) representa o valor base do imóvel quando a área do primeiro andar é igual a zero.

O coeficiente 6793.64 indica que, para cada unidade adicional de área, o preço aumenta, em média, R$ 6.793,64.

## Avaliação do modelo

O R² no conjunto de treino foi 0.377. Isso significa que 37,7% da variação do preço é explicada apenas pela área do primeiro andar.

No conjunto de teste, o R² foi:

```python
from sklearn.metrics import r2_score

y_predict = modelo_0.predict(X_test)
r2_score(y_test, y_predict)
```

R² = 0.385

O desempenho semelhante entre treino e teste indica que o modelo apresenta comportamento consistente e não há evidência forte de overfitting.

## Resíduos

Os resíduos representam a diferença entre o valor real e o valor previsto pelo modelo:

resíduo = valor real − valor previsto

Resíduos positivos indicam que o modelo subestimou o preço.
Resíduos negativos indicam que o modelo superestimou.

A análise do histograma mostra dispersão considerável e presença de outliers, sugerindo que outras variáveis também influenciam o preço.

## Conclusão

A área do primeiro andar possui impacto positivo e estatisticamente significativo sobre o preço do imóvel. Entretanto, como o R² é moderado, o modelo pode ser aprimorado com a inclusão de outras variáveis explicativas.
