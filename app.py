import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils import pandas_agent as agent

print(agent("What was the highest/average/lowest stock price across for Month Jan-Apr"))
