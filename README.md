# IUST NLP project spring 2023
## Analyzing the Impact of News on Cryptocurrency Prices

### Description

This project aims to evaluate the impact of news events on cryptocurrency prices. The dataset used in this study was gathered from two prominent sources in the cryptocurrency industry: Cryptonews.com and Binance.com. The dataset includes information on various cryptocurrencies such as Bitcoin and Ethereum, as well as news articles related to these assets. By analyzing the data collected from these sources, the project seeks to provide insights into the relationship between news events and crypto market trends.

### Tools and Techniques

The project uses advanced analytical techniques to identify any correlations between news events and price movements. These techniques include sentiment analysis, natural language processing, and machine learning algorithms. The project is implemented using Python programming language and its data science libraries such as Pandas, NumPy, and Pytorch.

### Results

The project aims to provide valuable insights for investors and traders looking to make informed decisions in the cryptocurrency market. By understanding how news events can impact prices, market participants may be able to position themselves more effectively and potentially achieve greater returns on their investments.

## Execution Instructions
### Requirements

To use this program, you will need to have Python 3 installed on your machine. You will also need to set up a Python environment and install the packages listed in the 'requirements.txt' file. You can do this by running the following commands in a terminal or command prompt:

```
cd /path/to/root/of/repo
python -m venv venv
source venv/bin/activate   # On Linux or macOS. For Windows, run 'venv\Scripts\activate.bat'
python pip install -r requirements.txt
```

### Usage

To run the Python program, activate the virtual environment and run the following command from the root directory of the repository:

```
python main.py [--crawl] [--clean] [--add-label] [--analyse] [--labeling-method COLOR / ROC]
```

Replace `main.py` with the name of your Python file. The `--crawl`, `--clean`, `--add-label`, `--analyse`, and `--labeling-method` options are optional and can be used depending on the functionality you want to execute. You can also specify the desired labeling method using the `--labeling-method` option followed by either `COLOR` or `ROC`.

For example, to clean the data, you can run the command `python main.py --clean`. To add labels using the ROC method, you can run the command `python main.py --add-label --labeling-method ROC`.

### Conclusion

This Python program provides a flexible and customizable way to collect, clean, and analyze data from the web. By following the instructions provided in this README, you should be able to set up and run the program on your machine with ease.