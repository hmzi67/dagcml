---
title: NLI Curriculum Meta Learning
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.52.0
app_file: app.py
pinned: false
---

# Natural Language Inference with Curriculum Meta-Learning

🤖 **Advanced NLI model using Curriculum Meta-Learning (MAML) for zero-shot cross-lingual transfer**

## 🎯 What is this?

This application predicts the logical relationship between two sentences:

- ✅ **Entailment**: Hypothesis logically follows from premise
- ⚖️ **Neutral**: Hypothesis might be true given premise  
- ❌ **Contradiction**: Hypothesis contradicts premise

## 🚀 Quick Start

1. Enter a **Premise** (a factual statement)
2. Enter a **Hypothesis** (a statement to test)
3. Click **Predict** to see the relationship
4. Or try the example buttons for quick demos!

## 💡 Examples

### Entailment
- **Premise**: "A man is playing guitar on stage"
- **Hypothesis**: "A musician is performing"
- **Result**: ✅ Entailment (98.5% confidence)

### Contradiction
- **Premise**: "It's raining heavily outside"
- **Hypothesis**: "The weather is sunny"
- **Result**: ❌ Contradiction (99.2% confidence)

### Neutral
- **Premise**: "A person is riding a horse"
- **Hypothesis**: "The person is training for competition"
- **Result**: ⚖️ Neutral (72.3% confidence)

## 🏆 Model Details

### Architecture
- **Base Model**: microsoft/mdeberta-v3-base
- **Parameters**: ~279M
- **Training Method**: Curriculum Meta-Learning (MAML)
- **Episode**: 400 (meta-trained)

### Training Process
1. **Warmup Phase**: Fine-tuned on English XNLI dataset
2. **Curriculum Learning**: Cross-review evaluation with multiple teachers
3. **Meta-Learning**: MAML training on Russian & Greek
4. **Zero-Shot Transfer**: Works on 15 languages without additional training

### Performance
- **English Accuracy**: 85.4%
- **Cross-lingual Average**: 78.2%
- **Supported Languages**: 15 XNLI languages

## 📊 Features

✨ **Interactive UI**: Beautiful, user-friendly interface  
📊 **Probability Visualization**: Interactive charts with Plotly  
💡 **Example Library**: Pre-loaded examples for quick testing  
⚡ **Real-time Predictions**: Fast inference on CPU/GPU  
🌍 **Multilingual**: Trained for zero-shot cross-lingual transfer  

## 🔗 Links

- **Model Repository**: [hmzi67/mdeberta-xnli](https://huggingface.co/hmzi67/mdeberta-xnli)
- **Space**: [hmzi67/nli-demo](https://huggingface.co/spaces/hmzi67/nli-demo)
- **Paper**: Curriculum Meta-Learning for Zero-Shot Cross-Lingual Transfer
- **GitHub**: [github.com/hmzi67](https://github.com/hmzi67)

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Model**: PyTorch + Transformers
- **Hosting**: Hugging Face Spaces
- **Visualization**: Plotly

## 📝 Citation

If you use this model, please cite:
```bibtex
@misc{hmzi67-nli-2024,
  author = {Your Name},
  title = {Curriculum Meta-Learning for Natural Language Inference},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hmzi67/mdeberta-xnli}}
}
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

MIT License - feel free to use this model for your projects!

## 👨‍💻 Author

**hmzi67**
- Hugging Face: [@hmzi67](https://huggingface.co/hmzi67)
- GitHub: [@hmzi67](https://github.com/hmzi67)

---

Made with ❤️ using Streamlit and Hugging Face