# trw_ta

**`trw_ta`** is a Python library focused on converting TradingView Pine Script indicators into native Python for quantitative analysis and backtesting.  
It is developed and maintained for **The Real World** community to enable research and integration of Pine Script strategies into Python-based systems.

## ✨ Features

- One-to-one conversions of TradingView indicators to pandas-compatible Python functions.
- Clean, modular architecture for technical analysis workflows.
- Open-source for community contributions and expansion.

## 📦 Installation

This is an early development release. You can install directly from GitHub:

```bash
pip install git+https://github.com/<your-username>/trw_ta.git
````

### 🔧 OS-specific notes:

#### ✅ Windows (CMD or PowerShell):

Make sure `git` and `pip` are in your PATH. Then run:

```bash
pip install git+https://github.com/<your-username>/trw_ta.git
```

#### ✅ macOS (Terminal):

```bash
pip install git+https://github.com/<your-username>/trw_ta.git
```

#### ✅ Linux (Ubuntu/Debian):

If `pip` is not installed, first run:

```bash
sudo apt update && sudo apt install python3-pip git
```

Then install:

```bash
pip install git+https://github.com/<your-username>/trw_ta.git
```

---

## 📚 Usage

```python
import trw_ta as ta

# Example: Use a translated TradingView indicator
df['RSI'] = ta.rsi(df['close'], 14)
```

Most functions accept `pandas.Series` inputs for flexibility and integration with your backtesting pipeline.

---

## 💡 Contributing

This project is in early-stage development. If you're a member of **The Real World** or interested in Pine Script conversion, feel free to fork, extend, or contribute.
