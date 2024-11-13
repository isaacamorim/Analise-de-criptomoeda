import yfinance as yf   # Usado para baixar dados financeiros históricos.
import pandas as pd     # Usado para manipulação e análise de dados.
import talib    # Uma biblioteca de análise técnica usada para calcular vários indicadores.
import matplotlib.pyplot as plt # Usado para criar visualizações.
import numpy as np
import seaborn as sns
import time

#Obter dados históricos do Bitcoin e do S&P 500
btc = yf.download('BTC-USD', start='2022-01-01', end='2024-10-30')
sp500 = yf.download('^GSPC', start='2022-01-01', end='2024-10-30')


# Calcular indicadores
btc['RSI'] = talib.RSI(btc['Close'], timeperiod=14)     # RSI (Índice de Força Relativa): Mede o momentum das mudanças de preço.
btc['MACD'], btc['MACDsignal'], btc['MACDhist'] = talib.MACD(btc['Close'], fastperiod=12, slowperiod=26, signalperiod=9)    # MACD (Convergência e Divergência da Média Móvel): Identifica a força da tendência e reversões potenciais.
btc['%K'], btc['%D'] = talib.STOCH(btc['High'], btc['Low'], btc['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
upper, middle, lower = talib.BBANDS(btc['Close'], timeperiod=20)
btc['upper'] = upper
btc['middle'] = middle
btc['lower'] = lower
btc['WilliamsR'] = talib.WILLR(btc['High'], btc['Low'], btc['Close'], timeperiod=14)


# Definir regras de compra e venda (exemplo)
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['signal'][data['RSI'] < 30] = 1.0  # Sinal de compra
    signals['signal'][data['RSI'] > 70] = -1.0  # Sinal de venda
    return signals

# Gerar os sinais
signals = generate_signals(btc)

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(btc['Close'], label='Preço de Fechamento')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(btc['RSI'], label='RSI')
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='r', linestyle='--')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(btc['MACD'], label='MACD')
plt.plot(btc['MACDsignal'], label='Sinal')
plt.bar(btc.index, btc['MACDhist'], label='Histograma')
plt.legend()

plt.tight_layout()
plt.show()

# Definir stop-loss e take-profit (simplificado)
stop_loss = 0.95  # 5% abaixo do preço de entrada
take_profit = 1.1  # 10% acima do preço de entrada

# Função para calcular o retorno total de uma estratégia
# Função para calcular o retorno total de uma estratégia
def calculate_return(data, signals, stop_loss, take_profit):
    positions = signals.diff()
    portfolio = initial_capital = 10000
    in_position = False
    for i in range(len(data)):
        if positions.iloc[i] == 1:
            if not in_position:  # Só compra se não estivermos em posição
                in_position = True
                buy_price = data['Close'].iloc[i]
                stop_loss_price = buy_price * stop_loss
                take_profit_price = buy_price * take_profit
                print(f"Compra a {buy_price} com stop-loss em {stop_loss_price} e take-profit em {take_profit_price}")
        elif positions.iloc[i] == -1:
            if in_position:  # Só vende se estivermos em posição
                in_position = False
                sell_price = data['Close'].iloc[i]
                profit_loss = sell_price - buy_price
                portfolio += profit_loss
                print(f"Venda a {sell_price} com lucro/prejuízo de {profit_loss}")
        if in_position:
            if data['Close'].iloc[i] <= stop_loss_price:
                # Stop-loss atingido
                sell_price = stop_loss_price
                in_position = False
                profit_loss = sell_price - buy_price
                portfolio += profit_loss
                print(f"Stop-loss atingido. Venda a {sell_price} com lucro/prejuízo de {profit_loss}")
            elif data['Close'].iloc[i] >= take_profit_price:
                # Take-profit atingido
                sell_price = take_profit_price
                in_position = False
                profit_loss = sell_price - buy_price
                portfolio += profit_loss
                print(f"Take-profit atingido. Venda a {sell_price} com lucro/prejuízo de {profit_loss}")
        portfolio = portfolio + positions.iloc[i] * data['Close'].iloc[i]
    return (portfolio / initial_capital) - 1

# Função para otimizar a estratégia
def optimize_strategy(data, signals, stop_loss_range, take_profit_range):
    best_return = -np.inf
    best_params = None
    for stop_loss in stop_loss_range:
        for take_profit in take_profit_range:
            return_ = calculate_return(data, signals, stop_loss, take_profit)
            if return_ > best_return:
                best_return = return_
                best_params = (stop_loss, take_profit)
    return best_params, best_return

# Definir os intervalos dos parâmetros
stop_loss_range = np.arange(0.9, 0.95, 0.01)
take_profit_range = np.arange(1.05, 1.15, 0.01)

# Criar uma matriz para armazenar os resultados
results = np.zeros((len(stop_loss_range), len(take_profit_range)))

# Loop para testar todas as combinações de parâmetros
for i, stop_loss in enumerate(stop_loss_range):
    for j, take_profit in enumerate(take_profit_range):
        # Implementar a estratégia com os parâmetros atuais
        total_return = calculate_return(btc, signals, stop_loss, take_profit)
        # Atualizar a matriz de resultados
        results[i, j] = total_return


# Otimizar a estratégia
best_params, best_return = optimize_strategy(btc, signals, stop_loss_range, take_profit_range)
print("Melhores parâmetros:", best_params)
print("Melhor retorno:", best_return)

# Executar a estratégia com os melhores parâmetros
best_stop_loss, best_take_profit = best_params
final_return = calculate_return(btc, signals, best_stop_loss, best_take_profit)
print("Retorno final com os melhores parâmetros:", final_return)

# Otimizar a estratégia
results = optimize_strategy(btc, signals, stop_loss_range, take_profit_range)

# Encontrar a melhor combinação de parâmetros
best_params = results.loc[results['return'].idxmax()]
print("Melhores parâmetros:", best_params)

# Visualizar os resultados
sns.heatmap(results.pivot('stop_loss', 'take_profit', 'return'))
plt.show()

# Otimizar a estratégia (já calculada no loop)
# best_params, best_return = optimize_strategy(btc, signals, stop_loss_range, take_profit_range)

# En
s
# Baixar dados do benchmark
sp500 = yf.download('^GSPC', start='2022-01-01', end='2024-10-30')

# Calcular o retorno do S&P 500
sp500_return = (sp500['Close'].iloc[-1] / sp500['Close'].iloc[0]) - 1

# Plotar a curva de equidade da estratégia e do benchmark
plt.figure(figsize=(12, 6))
plt.plot(btc.index, portfolio_values, label='Estratégia')
plt.plot(sp500.index, sp500['Close'], label='S&P 500')
plt.title('Comparação da Estratégia com o S&P 500')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()



python seu_script.py > log.txt

input("Pressione Enter para sair...")
time.sleep(5)  # Aguarda 5 segundos antes de fechar
