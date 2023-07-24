import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
# import pytesseract
from glob import glob
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
from tqdm import tqdm
import torch.optim as optim
from unidecode import unidecode
from sklearn.model_selection import train_test_split
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')