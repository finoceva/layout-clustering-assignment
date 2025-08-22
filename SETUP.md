# Quick Setup Guide

## 📋 **Prerequisites**

- Python 3.11+
- OpenAI API key (for AI recommendations)

## 🚀 **Installation**

1. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API:**
   ```bash
   # Copy environment template
   cp env.example .env

   # Edit .env and add your API key:
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Verify installation:**
   ```bash
   python -c "import recommendation.engine; print('✅ Installation successful')"
   ```

## 🎯 **Usage**

### **Complete Analysis**
```bash
python main.py
```

### **Interactive Demo**
```bash
python demo.py
```

### **Configuration**
- **Default settings**: `config/recommendation_config.yaml`
- **Environment overrides**: `OPENAI_MODEL=gpt-4 python main.py`

### **Testing**
```bash
pytest
```

## 🔧 **Configuration Options**

| Setting | Default | Override |
|---------|---------|----------|
| OpenAI Model | `gpt-4.1-mini` | `OPENAI_MODEL=gpt-4` |
| Temperature | `0.7` | `OPENAI_TEMPERATURE=0.3` |
| Max Tokens | `300` | `OPENAI_MAX_TOKENS=500` |

## 📁 **Key Files**

- `main.py` - Main entry point
- `demo.py` - Interactive demonstration
- `config/recommendation_config.yaml` - Main configuration
- `data/01_raw/assignment_data.json` - Input dataset

## ❓ **Troubleshooting**

**No OpenAI API key:**
```
ValueError: OPENAI_API_KEY environment variable is required
```
→ Add your API key to `.env` file

**Import errors:**
```
ModuleNotFoundError: No module named 'transformers'
```
→ Run `uv sync` or `pip install -r requirements.txt`

**Configuration errors:**
```
logger.error: Configuration validation failed
```
→ Check `config/recommendation_config.yaml` syntax

---

**🎯 You're ready to go! Run `python main.py` to start.**
