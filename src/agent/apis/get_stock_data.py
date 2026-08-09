import yfinance as yf

def get_stock_data(stock: str, period: str = "1mo", interval: str = "1d", include_info: bool = False):
    """
    Fetches stock market data. Period can be 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    Interval can be 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
    """
    try:
        stock_info = yf.Ticker(stock)
        # Fetching historical price data
        hist = stock_info.history(period=period, interval=interval)
        
        if hist.empty:
            return {"status_code": 404, "error": "Ticker not found or no data available."}

        # Core Price Data
        latest_price = hist['Close'].iloc[-1]
        opening_price = hist['Open'].iloc[-1]
        high_price = hist['High'].max()
        
        data = {
            "symbol": stock.upper(),
            "current_price": round(latest_price, 2),
            "currency": "USD", # yf usually defaults to USD for US stocks
            "change_pct": round(((latest_price - opening_price) / opening_price) * 100, 2),
            "period_high": round(high_price, 2)
        }

        # Adding "Challenging" Metadata if requested
        if include_info:
            info = stock_info.info
            data["market_metadata"] = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector")
            }

        return {"status_code": 200, "data": data}
    except Exception as e:
        return {"status_code": 500, "error": str(e)}

if __name__ == "__main__":
    ticker_symbol = "NTDOY"
    stock_info = get_stock_data(ticker_symbol, period="1d", interval="1d", include_info=True)
    print(stock_info)