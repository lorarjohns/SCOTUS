color_dict = {}        
for k,v in ddf.items():
    for k in ddf:
        rgb = np.random.rand(3,)
        color_dict[matplotlib.colors.rgb2hex(rgb)] = ddf[k]

from random import random
from wordcloud import WordCloud, get_single_color_func

class Grouped:
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    
    def __init__(self, cdict, default_color):
        self.cdict = cdict
        self.color_func_to_words = [(get_single_color_func(color), words)
                                   for (color, words) in self.cdict.items()]

        self.default_color_func = get_single_color_func(default_color)
        
    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration as e:
            print(f"Could not apply custom color func: {e}")
            color_func = self.default_color_func
            
        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)  
    
def build_color_dict(item_dict):
    color_dict = {}        
    for k,v in item_dict.items():
        for k in item_dict:
            rgb = np.random.rand(3,)
            color_dict[matplotlib.colors.rgb2hex(rgb)] = item_dict[k]
    return color_dict
    
makecloud("law.png")
#default_color = "gray"
#color_shades = Grouped(color_dict, default_color)

#wc.recolor(color_func=color_shades) # contour_color="black", contour_width=3,


import os
import numpy as np
import csv
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import cv2 
import imutils

def make_mask(image_path):
    mask = Image.open(image_path)
    mask = mask.convert("1", dither=Image.NONE)
    return np.array(mask)

def make_cloud(image_path): # csv_path, image_path

    mask = make_mask(image_path)
    csv_path = "wordcloud.csv"
    font_path = "/System/Library/Fonts/Helvetica.ttc"  
    
    #AmericanTypewriter.ttc"
    
    message = defaultdict(float)
    
    with open(csv_path, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message[row["word"]] = int(row["count"])
    
    message = dict(message)
    
    wc = WordCloud(font_path=font_path, mask=mask, mode="RGBA", background_color=None) 
    #mask=mask, mode="RGB", contour_color="lightgreen", contour_width=3)

    # generate word cloud
    wc.generate_from_frequencies(message)
    #color_shades = Grouped(color_dict, default_color)
    #wc.recolor(color_func=color_shades)

    wc.to_file("wordcloud.png")

    plt.figure()
    plt.imshow(mask)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()


    def build_color_dict(item_dict):
    color_dict = {}        
    for k,v in item_dict.items():
        for k in item_dict:
            rgb = np.random.rand(3,)
            color_dict[matplotlib.colors.rgb2hex(rgb)] = item_dict[k]
    return color_dict


    def make_time_clouds(data, csv_path=None, font_paths=None, color_func=None, imgs=None):
    
    imgs = []
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    contour_color = "black"
    
    if imgs is not None:
        masks = [np.array(Image.read(path)) for img in imgs]
    
    color_dict = build_color_dict(data)
    
    with open(csv_path, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message[row["term"]] = float(row["weight"])
    
    for key in data:
        if masks  == []:
            wc = WordCloud(ranks_only=True, font_path=font, max_words=1000,
                           mode="RGB", contour_color=contour_color, max_font_size=150, 
                           contour_width=3, random_state=42)
        
        color_shades = Grouped(color_dict, default_color)
                                    
        # wc = WordCloud(ranks_only=True, width=1000, height=1000, max_words=200, max_font_size=150,
        #              collocations=False, prefer_horizontal=-2) #, #mask=mask, mode="RGB", contour_color="lightgreen", contour_width=3)

        # generate word clouds
        wc.generate_from_frequencies(message)
        default_color = "gray"
        color_shades = Grouped(color_dict, default_color)
        wc.recolor(color_func=color_shades)
    
        wc.to_file(f"cloud+_+{k}.png")

        plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.show()

        def make_time_clouds(data, csv_path=None, font_paths=None, color_func=None, imgs=None):
    
    imgs = []
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    contour_color = "black"
    
    if imgs is not None:
        masks = [np.array(Image.read(path)) for img in imgs]
    
    color_dict = build_color_dict(data)
    
    with open(csv_path, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message[row["term"]] = float(row["weight"])
    
    for key in data:
        if masks  == []:
            wc = WordCloud(ranks_only=True, font_path=font, max_words=1000,
                           mode="RGB", contour_color=contour_color, max_font_size=150, 
                           contour_width=3, random_state=42)
        
        color_shades = Grouped(color_dict, default_color)
                                    
        # wc = WordCloud(ranks_only=True, width=1000, height=1000, max_words=200, max_font_size=150,
        #              collocations=False, prefer_horizontal=-2) #, #mask=mask, mode="RGB", contour_color="lightgreen", contour_width=3)

        # generate word clouds
        wc.generate_from_frequencies(message)
        default_color = "gray"
        color_shades = Grouped(color_dict, default_color)
        wc.recolor(color_func=color_shades)
    
        wc.to_file(f"cloud+_+{k}.png")

        plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.show()

def make_time_clouds(data, csv_path=None, font_paths=None, color_func=None, imgs=None):
    
    imgs = []
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    contour_color = "black"
    
    if imgs is not None:
        masks = [np.array(Image.read(path)) for img in imgs]
    
    color_dict = build_color_dict(data)
    
    with open(csv_path, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message[row["term"]] = float(row["weight"])
    
    for key in data:
        if masks  == []:
            wc = WordCloud(ranks_only=True, font_path=font, max_words=1000,
                           mode="RGB", contour_color=contour_color, max_font_size=150, 
                           contour_width=3, random_state=42)
        
        color_shades = Grouped(color_dict, default_color)
                                    
        # wc = WordCloud(ranks_only=True, width=1000, height=1000, max_words=200, max_font_size=150,
        #              collocations=False, prefer_horizontal=-2) #, #mask=mask, mode="RGB", contour_color="lightgreen", contour_width=3)

        # generate word clouds
        wc.generate_from_frequencies(message)
        default_color = "gray"
        color_shades = Grouped(color_dict, default_color)
        wc.recolor(color_func=color_shades)
    
        wc.to_file(f"cloud+_+{k}.png")

        plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.show()


wc.layout_ 

'''
``layout_`` : list of tuples (string, int, (int, int), int, color))
    Encodes the fitted word cloud. Encodes for each word the string, font
    size, position, orientation and color.
     
'''