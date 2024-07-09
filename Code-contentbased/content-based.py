import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import webbrowser

#FRAMES
root = tk.Tk()
root.geometry("1100x1000")

#LOAD DATA + IMAGES
DKdf_model = pd.read_csv('Data_output/DKdf_LinearSVR.csv', delimiter=',')
Full_df = pd.read_csv('Data_output/DHLdf_cosine.csv', delimiter=',')

home_image = Image.open("..//pictures//dashboard.jpg")
home_icon = ImageTk.PhotoImage(home_image.resize((40, 25), Image.Resampling.LANCZOS))
favorites_image = Image.open("..//pictures//favorites.jpg")
favorites_icon = ImageTk.PhotoImage(favorites_image.resize((50, 25), Image.Resampling.LANCZOS))
info_image = Image.open("..//pictures//info.jpg")
info_icon = ImageTk.PhotoImage(info_image.resize((35, 25), Image.Resampling.LANCZOS))
account_image = Image.open("..//pictures//account.png")
account_icon = ImageTk.PhotoImage(account_image.resize((25, 25), Image.Resampling.LANCZOS))
irene_image = Image.open("..//pictures//irene.jpg")
irene_photo = ImageTk.PhotoImage(irene_image.resize((250, 300)))
food_image = Image.open("..//pictures//food.jpg")
food_photo = ImageTk.PhotoImage(food_image.resize((120, 100)))

#STYLE
light_framecolor = "#d8f3ac"
framecolor = "#77b318"
lettertype = "Bahnschrift"

#OUTPUT: show stockphoto, recipe, serves, minutes, save to Favorites, go to recipe
def submit_text(rec_incl, max_scores, url, Full_df):
    for idx, (rec, score, u) in enumerate(zip(rec_incl, max_scores, url)):
        recipe_frame = tk.Frame(output_frame, bg=light_framecolor, pady=10, padx=10, highlightbackground="black", highlightthickness=1)
        row, col = divmod(idx, 3)
        recipe_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")


        food_label = tk.Label(recipe_frame, image=food_photo, bg="white", highlightbackground="black", highlightthickness=1)
        food_label.image = food_photo 
        food_label.pack(side="top", padx=10, pady=10)

        row_data = Full_df.loc[Full_df['title'] == rec]
        serves = int(row_data['serveslist'].iloc[0])
        time = int(row_data['cookingtimelist'].iloc[0])

        rec_label = tk.Label(recipe_frame, text=f"{rec}\nScore: {score}\n{time} min. Serves: {serves}", bg=light_framecolor, fg="black", font=(lettertype, 10), justify="left")
        rec_label.pack(side="top", padx=10)
        
        save_button = tk.Button(recipe_frame, text="Save to Favorites", command=lambda: save_favorites(rec, score, u), bg="white", fg="#006400", font=(lettertype, 10, "bold"))
        save_button.pack(side="top", padx=10, pady=5)

        url_button = tk.Button(recipe_frame, text="Go to Recipe", command=lambda: webbrowser.open(u), bg=framecolor, fg="white", font=(lettertype, 10, "bold"))
        url_button.pack(side="top", padx=10, pady=5)

    for i in range(2): 
        output_frame.grid_rowconfigure(i, weight=1)
    for i in range(3):
        output_frame.grid_columnconfigure(i, weight=1)


#MENU'S: ervoor zorgen dat ze elkaar niet overlappen
def show_home_page():
    hide_all_frames()
    home_frame.pack(fill="both", expand=1)
def show_favorites():
    hide_all_frames()
    favorites_frame.pack(fill="both", expand=1)
def show_info():
    hide_all_frames()
    info_frame.pack(fill="both", expand=1)
def show_account():
    hide_all_frames()
    account_frame.pack(fill="both", expand=1)
def hide_all_frames():
    home_frame.pack_forget()
    favorites_frame.pack_forget()
    info_frame.pack_forget()
    account_frame.pack_forget()

#INSIDE MODEL
def get_recommendations(cosine_sim, num_recommend, DKdf_model):
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:num_recommend]
    recipe_indices = [i[0] for i in sim_scores]
    return DKdf_model.iloc[recipe_indices]['Recipename'].values

def inside_model(DKdf_model, input_text, randomness, preferences, allergies):

    #Get the top 20 most similar recipes
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(DKdf_model['Ingredients'])
    tfidf_input = tfidf.transform([input_text])
    cosine_sim = linear_kernel(tfidf_input, tfidf_matrix)
    rec_excl = get_recommendations(cosine_sim, num_recommend=20, DKdf_model=DKdf_model)
    rows = DKdf_model[DKdf_model['Recipename'].isin(rec_excl)].copy()

    #Filter by preferences
    if preferences == "Vegetarian":
        DKdf_model = DKdf_model[DKdf_model['Vegetarian'] == 1]
    elif preferences == "Vegan":
        DKdf_model = DKdf_model[DKdf_model['Vegan'] == 1]
    else:
        DKdf_model = DKdf_model

    #Filter by allergies
    allergy_dict = {
        'Milk': ['milk', 'butter', 'butter fat', 'cheese', 'cottage cheese', 'cheese sauce', 'cream', 'sour cream', 'custard', 'buttermilk', 'powdered milk', 'evaporated milk', 'yogurt', 'ice cream', 'pudding'],
        'Eggs': ['egg', 'cakes', 'cookies', 'pastries', 'mayonnaise', 'meringue', 'custard', 'pudding', 'hollandaise sauce', 'quiche', 'omelets', 'frittata'],
        'Wheat': ['wheat', 'bread', 'pasta', 'cereal', 'crackers', 'flour', 'pizza dough', 'pancakes', 'waffles', 'muffins', 'pastries', 'cookies', 'cakes', 'pretzels', 'tortillas', 'beer'],
        'Soy': ['soy', 'tofu', 'miso', 'tempeh', 'edamame', 'natto'],
        'Tree_nuts': ['peanuts', 'almonds', 'cashews', 'walnuts', 'pecans', 'hazelnuts', 'pistachios', 'macadamia nuts', 'brazil nuts', 'pine nuts', 'nut butters', 'nut oils', 'marzipan', 'baklava', 'brittle'],
        'Fish': ['fish', 'salmon', 'tuna', 'cod', 'haddock', 'anchovies', 'sardines', 'mackerel', 'tilapia', 'swordfish', 'fish sauce', 'fish sticks', 'fish oil', 'canned fish'],
        'Shellfish': ['shellfish', 'shrimp', 'crab', 'lobster', 'scallops', 'oysters', 'mussels', 'clams', 'prawns', 'crayfish', 'shrimp sauce', 'clam chowder'],
        'Sesame': ['sesame', 'tahini', 'hummus', 'goma-dare']
    }

    if allergies in allergy_dict:
        allergy_ingredients = allergy_dict[allergies]
        pattern = '|'.join([r'\b' + ingredient + r'\b' for ingredient in allergy_ingredients])
        DKdf_model = DKdf_model[~DKdf_model['Ingredients'].str.contains(pattern, case=False, na=False)]

    #Determine recommendations based on randomness slider
    if randomness == 1:
        max_score_indices = rows['Predicted_Score'].nlargest(6).index
    elif randomness == 0:
        cosine_sim_scores = linear_kernel(tfidf_input, tfidf_matrix[rows.index])
        rows.loc[:, 'cosine_sim'] = cosine_sim_scores[0]
        rows = rows.sort_values(by='cosine_sim', ascending=False)
        max_score_indices = rows.head(6).index
    else:
        recommendation_indices = rows['Predicted_Score'].nlargest(6).index
        cosine_sim_scores = linear_kernel(tfidf_input, tfidf_matrix[rows.index])
        rows.loc[:, 'cosine_sim'] = cosine_sim_scores[0]
        rows = rows.sort_values(by='cosine_sim', ascending=False)  

        weight_recommendation = randomness
        recommended = int(6 * weight_recommendation)
        similarity = 6-recommended
        blended_indices = np.concatenate([recommendation_indices[:recommended], rows.head(similarity).index])
        max_score_indices = np.unique(blended_indices)

    rec_incl = rows.loc[max_score_indices]['Recipename'].tolist()
    scores = rows.loc[max_score_indices]['Predicted_Score'].tolist()
    url = rows.loc[max_score_indices]['url'].tolist()

    #Assign ABCD scores
    max_scores = []
    quantiles = DKdf_model['Predicted_Score'].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = quantiles[0.25], quantiles[0.5], quantiles[0.75]

    for score in scores:
        if score <= q1:
            max_scores.append('D')
        elif score <= q2:
            max_scores.append('C')
        elif score <= q3:
            max_scores.append('B')
        else:
            max_scores.append('A')

    return rec_incl, max_scores, url

def submit_button_click():
    input_text = text_entry.get()
    preferences = preferences_var.get()
    allergies = allergies_var.get()
    if input_text.strip() == "":
        error_label.config(text="Please enter some text.")
    else:
        randomness = randomness_slider.get()
        rec_incl, max_scores, url = inside_model(DKdf_model, input_text, randomness, preferences, allergies)
        submit_text(rec_incl, max_scores, url, Full_df)



#SIDEBAR FRAME
sidebar = tk.Frame(root, bg=framecolor)
sidebar.pack(side="left", fill="y")
sidebar_title = tk.Label(sidebar, text="Menu", bg=framecolor, fg="black", font=(lettertype, 16, "bold"))
sidebar_title.pack(pady=20)

home_button = tk.Button(sidebar, text="Home Page", image=home_icon, compound="left", command=show_home_page, bg="white", font=(lettertype, 10, "bold"), width=180)
home_button.pack(pady=10)
favorites_button = tk.Button(sidebar, text="Favorites     ", image=favorites_icon, compound="left", command=show_favorites, bg="white", font=(lettertype, 10, "bold"), width=180)
favorites_button.pack(pady=10)
info_button = tk.Button(sidebar, text="Information ", image=info_icon, compound="left", command=show_info, bg="white", font=(lettertype, 10, "bold"), width=180)
info_button.pack(pady=10)
account_button = tk.Button(sidebar, text="   Account    ", image=account_icon, compound="left", command=show_account, bg="white", font=(lettertype, 10, "bold"), width=180)
account_button.pack(pady=10)

#BASIC FRAMES
start_frame = tk.Frame(root)
start_frame.pack(side="left", fill="both", expand=True)

home_frame = tk.Frame(start_frame, bg="white")
favorites_frame = tk.Frame(start_frame, bg="white")
info_frame = tk.Frame(start_frame, bg="white")
account_frame = tk.Frame(start_frame, bg="white")

#HOMEPAGE
style = ttk.Style()
style.theme_use('clam')
style.configure('Preferences.TMenubutton', background=light_framecolor, foreground="black", font=(lettertype, 12))
style.configure('Preferences.Horizontal.TScale', background=light_framecolor, troughcolor=framecolor, sliderlength=20)

title_label = tk.Label(home_frame, text="Sustainable Meal Recommender", font=(lettertype, 16, "bold"), bg=framecolor, fg="white")
title_label.pack(pady=10, fill=tk.X)

text_entry_label = tk.Label(home_frame, text="What ingredients do you have?", background="white", font=(lettertype, 12))
text_entry_label.pack(pady=5)
text_entry = tk.Entry(home_frame, width=50, font=(lettertype, 12), bg=light_framecolor)
text_entry.pack(pady=5)

#Frame for the slider and menus
options_frame = tk.Frame(home_frame, bg="white")
options_frame.pack(pady=5, fill=tk.X)

#Randomness slider
randomness_label = tk.Label(options_frame, text="Similarities (0) vs Recommended (1)", background="white", font=(lettertype, 12))
randomness_label.grid(row=0, column=0, padx=5, pady=5)
randomness_slider = tk.Scale(options_frame, from_=0, to=1, resolution=0.2, orient=tk.HORIZONTAL, length=200, sliderlength=20, bg=light_framecolor, troughcolor=light_framecolor)
randomness_slider.set(0)
randomness_slider.grid(row=0, column=1, padx=5, pady=5)

#Preferences menu
preferences_label = tk.Label(options_frame, text="Preferences:", background="white", font=(lettertype, 12))
preferences_label.grid(row=0, column=2, padx=5, pady=5)
preferences_var = tk.StringVar(options_frame)
preferences_var.set("None")
preferences_menu = tk.OptionMenu(options_frame, preferences_var, "None", "Vegan", "Vegetarian")
preferences_menu.config(bg=light_framecolor, font=(lettertype, 12))
preferences_menu.grid(row=0, column=3, padx=5, pady=5)

#Allergies menu
allergies_label = tk.Label(options_frame, text="Allergies:", background="white", font=(lettertype, 12))
allergies_label.grid(row=0, column=4, padx=5, pady=5)
allergies_var = tk.StringVar(options_frame)
allergies_var.set("None")
allergies_menu = tk.OptionMenu(options_frame, allergies_var, "None", "Milk", "Eggs", "Wheat", "Soy", "Tree nuts", "Fish", "Shellfish", "Sesame")
allergies_menu.config(bg=light_framecolor, font=(lettertype, 12))
allergies_menu.grid(row=0, column=5, padx=5, pady=5)

submit_button = tk.Button(home_frame, text="Submit", command=submit_button_click, bg=framecolor, fg="white", font=(lettertype, 12, "bold"))
submit_button.pack(pady=10)

output_frame = tk.Frame(home_frame, bg="white")
output_frame.pack(pady=10, fill="both", expand=True)

error_label = tk.Label(output_frame, text="", bg="white", fg="black")
error_label.grid(row=0, column=0)

#FAVORITES FRAME
favorites_list = []
title_label = tk.Label(favorites_frame, text="My Favorite Recipes", font=(lettertype, 16, "bold"), bg=framecolor, fg="white")
title_label.pack(pady=10, fill=tk.X)
recipe_container = tk.Frame(favorites_frame, bg="white")
recipe_container.pack(pady=10)


def save_favorites(recipe, score, url):
    favorites_list.append((recipe, score, url))
    update_favorites()

def update_favorites():
    for i, (rec, score, u) in enumerate(favorites_list):
        row_data = Full_df.loc[Full_df['title'] == rec]
        serves = int(row_data['serveslist'].iloc[0])
        time = int(row_data['cookingtimelist'].iloc[0])

        row_index = 1 + i // 4  
        col_index = i % 4    

        recipe_frame = tk.Frame(recipe_container, bg=light_framecolor, width=200, height=200, pady=10, padx=10, highlightbackground="black", highlightthickness=1)
        recipe_frame.grid(row=row_index, column=col_index, padx=10, pady=10)

        food_label = tk.Label(recipe_frame, image=food_photo, bg=light_framecolor, highlightbackground="black", highlightthickness=1)
        food_label.image = food_photo
        food_label.pack(side="top", padx=10, pady=10)

        fav_label = tk.Label(recipe_frame, text=f"{rec}\nScore: {score}\n{time} min. Serves: {serves}", bg=light_framecolor, fg="black", font=(lettertype, 10), justify="left")
        fav_label.pack(pady=5, fill="x", padx=10)

        url_button = tk.Button(recipe_frame, text="Go to Recipe", command=lambda u=u: webbrowser.open(u), bg=framecolor, fg="white", font=(lettertype, 10, "bold"))
        url_button.pack(pady=5)       

update_favorites()

#INFORMATION FRAME
info_label = tk.Label(info_frame, text="Information about the popularity score", font=(lettertype, 16, "bold"), bg=framecolor, fg="white")
info_label.pack(pady=10, fill=tk.X)

story_text = """Hi, I am Irene!

I have always been interested in cooking and, not to forget, eating food. 
Like many people, I want to contribute to saving the world by eating sustainably. 
However, I have had many conversations about what it means to eat sustainably. 
One person said that eating tofu is really good for the planet because it is a meat substitute. 
Another person said that eating tofu is really bad for the environment because it uses a lot of water. 
I am not the only one struggling with the challenge of eating sustainable and tasty food.

To help others, I created this dashboard that recommends food to reduce food waste by 
searching for sustainable and tasty recipes based on the food you have in your fridge.

The popularity score shown on the dashboard considers factors like rating, greenhouse gas emissions, 
land use, freshwater withdrawals, water use, and eutrophying emissions. The ratings of recipes indicate 
the starrating of previous users. Greenhouse gas emissions indicate the amount of gases released during 
the production of human activities, reflecting the environmental impact of recipes. Land use measures the 
total amount of land used to produce recipes. Freshwater withdrawals measure the volume of water extracted 
from freshwater sources, reflecting the recipe's water consumption. Stress-weighted water use considers varying 
levels of water scarcity, evaluating how the recipe impacts water stress. Eutrophying emissions quantify the release of 
nutrients into water bodies, which can lead to harmful algal blooms and toxic ecosystems, indicating the recipe's impact on water quality.

You can adjust the slider to your preference. If you prefer getting recipes recommended that are highly similar 
to your input ingredients, set the slider to 0. This can be useful if you don't want to go to the supermarket. 
If you prefer highly recommended recipes based on the popularity score, adjust the slider to 1. 
This can be useful if you have guests over and want to cook a sustainable and highly rated meal.

Hopefully you like the tool, and 'eet smakelijk!' 😊"""

story_label = tk.Label(info_frame, text=story_text, font=(lettertype, 10), bg="white", fg="black", justify=tk.CENTER)
story_label.pack(pady=10, fill=tk.X)

irene_label = tk.Label(info_frame, image=irene_photo, bg="white")
irene_label.image = irene_photo
irene_label.pack(pady=10)

#ACOUNT FRAME
account_label = tk.Label(account_frame, text="Account", font=(lettertype, 16, "bold"), bg=framecolor, fg="white")
account_label.pack(pady=10, fill=tk.X)

username_label = tk.Label(account_frame, text="Username:", font=(lettertype, 12, "bold") , bg="white")
username_label.pack()
username_entry = tk.Entry(account_frame, bg=light_framecolor)
username_entry.pack()

password_label = tk.Label(account_frame, text="Password:", font=(lettertype, 12, "bold"), bg="white")
password_label.pack()
password_entry = tk.Entry(account_frame, show="*", bg=light_framecolor)
password_entry.pack()

def login():
    print("Logging in...")

login_button = tk.Button(account_frame, text="Login", font=(lettertype, 12, "bold"), command=login, bg=framecolor, fg="white" )
login_button.pack(pady=20)

#START
show_home_page()
root.mainloop()
