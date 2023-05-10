import requests
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline


def get_stock_info():
    # 从东方财富网获取股票列表
    url = 'http://quote.eastmoney.com/stock_list.html'
    r = requests.get(url)
    r.encoding = 'gb2312'
    soup = BeautifulSoup(r.text, 'html.parser')
    stocks = []
    for ul in soup.find_all('ul'):
        for a in ul.find_all('a'):
            href = a.get('href')
            if href is not None and 'http://quote.eastmoney.com/sh' in href:
                stocks.append(href.split('/')[-1].split('.')[0])
            elif href is not None and 'http://quote.eastmoney.com/sz' in href:
                stocks.append(href.split('/')[-1].split('.')[0])

    # 根据股票列表逐个到百度股票获取个股详细信息数据
    stock_info = []
    for stock in stocks:
        url = f'https://gupiao.baidu.com/stock/{stock}.html'
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        data = {
            '股票代码': stock,
            '最新价': float(soup.select_one('.latest span').text),
            '涨跌额': float(soup.select_one('.price-change span').text),
            '涨跌幅': float(soup.select_one('.price-change b').text[:-1]),
            '今开': float(soup.select_one('.open b').text),
            '昨收': float(soup.select_one('.close b').text),
            '最高': float(soup.select_one('.high b').text),
            '最低': float(soup.select_one('.low b').text),
            '成交量': float(soup.select_one('.volume b').text[:-1]),
            '成交额': float(soup.select_one('.total-amount b').text[:-1])
        }
        stock_info.append(data)

    # 将结果存储到文件
    df = pd.DataFrame(stock_info)
    df.to_csv('stock_info_raw.csv', index=False)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def clean_data():
    # 加载数据集
    df = pd.read_csv('stock_info_raw.csv')

    # 删除缺失值
    df.dropna(inplace=True)

    # 删除异常值
    df = df[(df['最新价'] > 0) & (df['成交量'] > 0) & (df['成交额'] > 0)]

    # 删除重复值
    df.drop_duplicates(inplace=True)

    # 特
# 特征工程
# 计算涨跌幅
df['涨跌幅'] = df['涨跌幅'] / 100
# 计算股价相对于最高价的比例
df['股价相对最高价比例'] = df['最新价'] / df['最高']
# 计算股价相对于最低价的比例
df['股价相对最低价比例'] = df['最新价'] / df['最低']
# 计算成交量和成交额相对于均值的比例
df['成交量相对均值比例'] = df['成交量'] / df['成交量'].mean()
df['成交额相对均值比例'] = df['成交额'] / df['成交额'].mean()

# 选取需要用到的特征
features = ['涨跌额', '涨跌幅', '股价相对最高价比例', '股价相对最低价比例', '成交量相对均值比例', '成交额相对均值比例']
X = df[features].values
y = df['涨跌幅'].values

# 数据预处理
# 缩放和降维
pipeline = Pipeline([    ('scaler', MinMaxScaler()),    ('pca', PCA(n_components=5)),])
X = pipeline.fit_transform(X)

# 将处理后的数据存储到文件
df_cleaned = pd.DataFrame(data=X, columns=['特征1', '特征2', '特征3', '特征4', '特征5'])
df_cleaned['涨跌幅'] = y
df_cleaned.to_csv('stock_info_cleaned.csv', index=False)

return X, y
```python
if __name__ == '__main__':
    get_stock_info()
    X, y = clean_data()
import pandas as pd

# 读取股票数据
df = pd.read_csv('stock_data.csv')

# 进行数据清洗和特征工程
# ...

# 将特征和目标变量分离
X = df.drop(['Date', 'Close'], axis=1).values
y = df['Close'].values
from sklearn.preprocessing import StandardScaler

# 进行数据规范化
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 构建XGBoost模型
xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# 构建Keras模型
keras_model = Sequential()
keras_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
keras_model.add(Dropout(0.2))
keras_model.add(Dense(32, activation='relu'))
keras_model.add(Dropout(0
# 构建集成学习模型
ensemble_model = VotingRegressor([('lr', lr_model), ('xgb', xgb_model), ('keras', keras_model)], weights=[1,2,3])
ensemble_model.fit(X_train, y_train)

# 构建LSTM模型
X_train_t = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_t = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])))
lstm_model.add(Dense(1, activation='linear'))
lstm_model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
lstm_model.fit(X_train_t, y_train, epochs=100, batch_size=32, validation_data=(X_test_t, y_test), callbacks=[early_stop])

4. 使用模型进行预测和评估

在模型构建完成后，我们可以使用测试集数据对模型进行预测，并使用均方误差和决定系数来评估模型的表现。

``` python
# 使用模型进行预测
lr_pred = lr_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
keras_pred = keras_model.predict(X_test).flatten()
ensemble_pred = ensemble_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test_t).flatten()

# 评估模型表现
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
keras_mse = mean_squared_error(y_test, keras_pred)
keras_r2 = r2_score(y_test, keras_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)
lstm_mse = mean_squared_error(y_test, lstm_pred)
lstm_r2 = r2_score(y_test, lstm_pred)

print(f"Linear Regression: MSE={lr_mse}, R2={lr_r2}")
print(f"XGBoost: MSE={xgb_mse}, R2={xgb_r2}")
print(f"Keras: MSE={keras_mse}, R2={keras_r2}")
print(f"Ensemble Model: MSE={ensemble_mse}, R2={ensemble_r2}")
print(f"LSTM Model: MSE={lstm_mse}, R2={lstm_r2}")
