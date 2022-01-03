#import all the necessary libraries and tools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#read the dataframes and set variables for each

ent_anime_df = pd.read_csv("entertainment_anime.csv")
ent_comicbooks_df = pd.read_csv("entertainment_comicbooks.csv")
ent_harrypotter_df = pd.read_csv("entertainment_harrypotter.csv")
ent_movies_df = pd.read_csv("entertainment_movies.csv")
ent_music_df = pd.read_csv("entertainment_music.csv")
ent_starwars_df = pd.read_csv("entertainment_starwars.csv")
game_dota_df = pd.read_csv("gaming_dota2.csv")
game_gaming_df = pd.read_csv("gaming_gaming.csv")
game_lol_df = pd.read_csv("gaming_leagueoflegends.csv")
game_minecraft_df= pd.read_csv("gaming_minecraft.csv")
game_pokemon_df = pd.read_csv("gaming_pokemon.csv")
game_skyrim_df = pd.read_csv("gaming_skyrim.csv")
game_starcraft_df = pd.read_csv("gaming_starcraft.csv")
game_tf2_df = pd.read_csv("gaming_tf2.csv")
hum_adviceanimals_df = pd.read_csv("humor_adviceanimals.csv")
hum_circlejerk_df = pd.read_csv("humor_circlejerk.csv")
hum_facepalm_df = pd.read_csv("humor_facepalm.csv")
hum_funny_df = pd.read_csv("humor_funny.csv")
hum_igthft_df = pd.read_csv("humor_imgoingtohellforthis.csv")
hum_jokes_df = pd.read_csv("humor_jokes.csv")
lrn_askhistorians_df = pd.read_csv("learning_askhistorians.csv")
lrn_askscience_df = pd.read_csv("learning_askscience.csv")
lrn_eli5_df = pd.read_csv("learning_explainlikeimfive.csv")
lrn_science_df = pd.read_csv("learning_science.csv")
lrn_space_df = pd.read_csv("learning_space.csv")
lrn_todayilearned_df = pd.read_csv("learning_todayilearned.csv")
lrn_youshouldknow_df = pd.read_csv("learning_youshouldknow.csv")
ls_drunk_df = pd.read_csv("lifestyle_drunk.csv")
ls_food_df = pd.read_csv("lifestyle_food.csv")
lf_frugal_df = pd.read_csv("lifestyle_frugal.csv")
lf_guns_df = pd.read_csv("lifestyle_guns.csv")
lf_lifehacks_df = pd.read_csv("lifestyle_lifehacks.csv")
lf_motorcycles_df = pd.read_csv("lifestyle_motorcycles.csv")
lf_progresspics_df = pd.read_csv("lifestyle_progresspics.csv")
lf_sex_df = pd.read_csv("lifestyle_sex.csv")
news_conservative_df = pd.read_csv("news_conservative.csv")
news_conspiracy_df = pd.read_csv("news_conspiracy.csv")
news_libertarian_df = pd.read_csv("news_libertarian.csv")
news_news_df = pd.read_csv("news_news.csv")
news_offbeat_df = pd.read_csv("news_offbeat.csv")
news_politics_df = pd.read_csv("news_politics.csv")
news_truereddit_df = pd.read_csv("news_truereddit.csv")
news_worldnews_df = pd.read_csv("news_worldnews.csv")
tv_breakingbad_df = pd.read_csv("television_breakingbad.csv")
tv_community_df = pd.read_csv("television_community.csv")
tv_doctorwho_df = pd.read_csv("television_doctorwho.csv")
tv_gameofthrones_df = pd.read_csv("television_gameofthrones.csv")
tv_himym_df = pd.read_csv("television_himym.csv")
tv_mylittlepony_df = pd.read_csv("television_mylittlepony.csv")
tv_startrek_df = pd.read_csv("television_startrek.csv")
tv_thewalkingdead_df = pd.read_csv("television_thewalkingdead.csv")

#create new column names since the original names aren't helpful
#headers: text,id,subreddit,meta,time,author,ups,downs,authorlinkkarma,authorkarma,authorisgold

ent_anime_df = ent_anime_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ent_comicbooks_df = ent_comicbooks_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ent_harrypotter_df = ent_harrypotter_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ent_movies_df = ent_movies_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ent_music_df = ent_music_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ent_starwars_df = ent_starwars_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_dota_df = game_dota_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_gaming_df = game_gaming_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_lol_df = game_lol_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_minecraft_df = game_minecraft_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_pokemon_df = game_pokemon_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_skyrim_df = game_skyrim_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_starcraft_df = game_starcraft_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
game_tf2_df = game_tf2_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_adviceanimals_df = hum_adviceanimals_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_circlejerk_df = hum_circlejerk_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_facepalm_df = hum_facepalm_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_funny_df = hum_funny_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_igthft_df = hum_igthft_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
hum_jokes_df = hum_jokes_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_askhistorians_df = lrn_askhistorians_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_askscience_df = lrn_askscience_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_eli5_df = lrn_eli5_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_science_df = lrn_science_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_todayilearned_df = lrn_todayilearned_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lrn_youshouldknow_df = lrn_youshouldknow_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ls_drunk_df = ls_drunk_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
ls_food_df = ls_food_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_frugal_df = lf_frugal_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_guns_df = lf_guns_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_lifehacks_df = lf_lifehacks_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_motorcycles_df = lf_motorcycles_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_progresspics_df = lf_progresspics_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
lf_sex_df = lf_sex_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_conservative_df = news_conservative_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_conspiracy_df = news_conspiracy_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_libertarian_df = news_libertarian_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_news_df = news_news_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_offbeat_df = news_offbeat_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_politics_df = news_politics_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_truereddit_df = news_truereddit_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
news_worldnews_df = news_worldnews_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_breakingbad_df = tv_breakingbad_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_community_df = tv_community_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_doctorwho_df = tv_doctorwho_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_gameofthrones_df = tv_gameofthrones_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_himym_df = tv_himym_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_mylittlepony_df = tv_mylittlepony_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_startrek_df = tv_startrek_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})
tv_thewalkingdead_df = tv_thewalkingdead_df.rename(columns={'Unnamed: 0' : 'row_num_int', '0' : 'text', '1' : 'id', '2' : 'subreddit', '3' : 'meta', '4' : 'time', '5' : 'author', '6' : 'ups', '7' : 'downs', '8' : 'authorlinkkarma', '9' : 'authorkarma', '10' : 'authorisgold'})

#allow every column to be displayed

pd.set_option("display.max_columns", None)

#remove majority of data values found

ent_anime_df = ent_anime_df[1:5001]
ent_comicbooks_df = ent_comicbooks_df[1:5001]
ent_harrypotter_df = ent_harrypotter_df[1:5001]
ent_movies_df = ent_movies_df[1:5001]
ent_music_df = ent_music_df[1:5001]
ent_starwars_df = ent_starwars_df[1:5001]
game_dota_df = game_dota_df[1:5001]
game_gaming_df = game_gaming_df[1:5001]
game_lol_df = game_lol_df[1:5001]
game_minecraft_df = game_minecraft_df[1:5001]
game_pokemon_df = game_pokemon_df[1:5001]
game_skyrim_df = game_skyrim_df[1:5001]
game_starcraft_df = game_starcraft_df[1:5001]
game_tf2_df = game_tf2_df[1:5001]
hum_adviceanimals_df = hum_adviceanimals_df[1:5001]
hum_circlejerk_df = hum_circlejerk_df[1:5001]
hum_facepalm_df = hum_facepalm_df[1:5001]
hum_funny_df = hum_funny_df[1:5001]
hum_igthft_df = hum_igthft_df[1:5001]
hum_jokes_df = hum_jokes_df[1:5001]
lrn_askhistorians_df = lrn_askhistorians_df[1:5001]
lrn_askscience_df = lrn_askscience_df[1:5001]
lrn_eli5_df = lrn_eli5_df[1:5001]
lrn_science_df = lrn_science_df[1:5001]
lrn_todayilearned_df = lrn_todayilearned_df[1:5001]
lrn_youshouldknow_df = lrn_youshouldknow_df[1:5001]
ls_drunk_df = ls_drunk_df[1:5001]
ls_food_df = ls_food_df[1:5001]
lf_frugal_df = lf_frugal_df[1:5001]
lf_guns_df = lf_guns_df[1:5001]
lf_lifehacks_df = lf_lifehacks_df[1:5001]
lf_motorcycles_df = lf_motorcycles_df[1:5001]
lf_progresspics_df = lf_progresspics_df[1:5001]
lf_sex_df = lf_sex_df[1:5001]
news_conservative_df = news_conservative_df[1:5001]
news_conspiracy_df = news_conspiracy_df[1:5001]
news_libertarian_df = news_libertarian_df[1:5001]
news_news_df = news_news_df[1:5001]
news_offbeat_df = news_offbeat_df[1:5001]
news_politics_df = news_politics_df[1:5001]
news_truereddit_df = news_truereddit_df[1:5001]
news_worldnews_df = news_worldnews_df[1:5001]
tv_breakingbad_df = tv_breakingbad_df[1:5001]
tv_community_df = tv_community_df[1:5001]
tv_doctorwho_df = tv_doctorwho_df[1:5001]
tv_gameofthrones_df = tv_gameofthrones_df[1:5001]
tv_himym_df = tv_himym_df[1:5001]
tv_mylittlepony_df = tv_mylittlepony_df[1:5001]
tv_startrek_df = tv_startrek_df[1:5001]
tv_thewalkingdead_df = tv_thewalkingdead_df[1:5001]

#drop the null values found in the dataframes

ent_anime_df = ent_anime_df.dropna()
ent_comicbooks_df = ent_comicbooks_df.dropna()
ent_harrypotter_df = ent_harrypotter_df.dropna()
ent_movies_df = ent_movies_df.dropna()
ent_music_df = ent_music_df.dropna()
ent_starwars_df = ent_starwars_df.dropna()
game_dota_df = game_dota_df.dropna()
game_gaming_df = game_gaming_df.dropna()
game_lol_df = game_lol_df.dropna()
game_minecraft_df = game_minecraft_df.dropna()
game_pokemon_df = game_pokemon_df.dropna()
game_skyrim_df = game_skyrim_df.dropna()
game_starcraft_df = game_starcraft_df.dropna()
game_tf2_df = game_tf2_df.dropna()
hum_adviceanimals_df = hum_adviceanimals_df.dropna()
hum_circlejerk_df = hum_circlejerk_df.dropna()
hum_facepalm_df = hum_facepalm_df.dropna()
hum_funny_df = hum_funny_df.dropna()
hum_igthft_df = hum_igthft_df.dropna()
hum_jokes_df = hum_jokes_df.dropna()
lrn_askhistorians_df = lrn_askhistorians_df.dropna()
lrn_askscience_df = lrn_askscience_df.dropna()
lrn_eli5_df = lrn_eli5_df.dropna()
lrn_science_df = lrn_science_df.dropna()
lrn_todayilearned_df = lrn_todayilearned_df.dropna()
lrn_youshouldknow_df = lrn_youshouldknow_df.dropna()
ls_drunk_df = ls_drunk_df.dropna()
ls_food_df = ls_food_df.dropna()
lf_frugal_df = lf_frugal_df.dropna()
lf_guns_df = lf_guns_df.dropna()
lf_lifehacks_df = lf_lifehacks_df.dropna()
lf_motorcycles_df = lf_motorcycles_df.dropna()
lf_progresspics_df = lf_progresspics_df.dropna()
lf_sex_df = lf_sex_df.dropna()
news_conservative_df = news_conservative_df.dropna()
news_conspiracy_df = news_conspiracy_df.dropna()
news_libertarian_df = news_libertarian_df.dropna()
news_news_df = news_news_df.dropna()
news_offbeat_df = news_offbeat_df.dropna()
news_politics_df = news_politics_df.dropna()
news_truereddit_df = news_truereddit_df.dropna()
news_worldnews_df = news_worldnews_df.dropna()
tv_breakingbad_df = tv_breakingbad_df.dropna()
tv_community_df = tv_community_df.dropna()
tv_doctorwho_df = tv_doctorwho_df.dropna()
tv_gameofthrones_df = tv_gameofthrones_df.dropna()
tv_himym_df = tv_himym_df.dropna()
tv_mylittlepony_df = tv_mylittlepony_df.dropna()
tv_startrek_df = tv_startrek_df.dropna()
tv_thewalkingdead_df = tv_thewalkingdead_df.dropna()

#function to create regplot based on karma for any data frame

def karma_regplot(x_data, y_data, df_name):

    plt.figure(figsize=(15,10))
    plt.grid(True)
    plt.title("Regplot for amount of upvotes over how much karma using the " + df_name + "SubReddit")
    sns.regplot(x = x_data, y = y_data)
    plt.show()

#display karma regplot using pokemon data frame

karma_regplot(game_pokemon_df["authorkarma"], game_pokemon_df["ups"], 'Pokemon')

#function to create regplot based on amount of text for any data frame

def text_regplot(x_data, y_data, df_name):

    plt.figure(figsize=(15,10))
    plt.grid(True)
    plt.title("Regplot for amount of upvotes over how much text in the post using the " + df_name + "Subreddit")
    sns.regplot(x = x_data, y = y_data)
    plt.show()

#display text regplot using pokemon data frame

text_regplot(game_pokemon_df["text"].str.len(), game_pokemon_df["ups"], " Pokemon")

#define a function to create the features variable based on the amount of text and author karma

def features(data_df):

    data_text = data_df["text"]

    text_char_num = data_text.str.len()

    array_y_value = data_df["authorkarma"]

    feature = np.array([text_char_num, array_y_value], dtype = object)

    return feature

features(game_pokemon_df)

#define a function to create the target variable based on upvotes

def targets(data_df):

    target = np.array(data_df["ups"], dtype = object)

    return target

targets(game_pokemon_df)

#define and organize the data used in the train_test_split

X = features(game_pokemon_df)
Y = targets(game_pokemon_df)
X = np.transpose(X)
Y = np.transpose(Y)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.25, train_size = 0.75, random_state = 42)

stdsc = StandardScaler()
xTrain = stdsc.fit_transform(xTrain)
xTest = stdsc.transform(xTest)

gbr = GradientBoostingRegressor()
gbr.fit(xTrain, yTrain)

#program to display the prediction training model

predictions = gbr.predict(xTest)
plt.scatter(yTest, predictions)
plt.style.use('seaborn-whitegrid')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('GradientRegressor')
plt.plot(np.arange(0,0.4, 0.01), np.arange(0, 0.4, 0.01), color = 'blue')
plt.grid(True)
plt.show()

def PredictionsUsingKarma(model, karmaCount, scaller):
    karma = karmaCount * np.ones(3344)
    text_amount = np.arange(1, 3345)

    print("karma", karma)
    print("text", text_amount)

    #defining vector
    featureVector = np.zeros((3344, 2))
    print("feature", featureVector)

    featureVector[:, 0] = karma
    print("feature vec", featureVector)

    featureVector[:, 1] = text_amount

    print("featurevector", featureVector)

    #doing scalling
    featureVector = scaller.transform(featureVector)
    predictions = model.predict(featureVector)
    predictions = (max(game_pokemon_df["ups"]) * predictions).astype('int')

    #program to plot the upvote progression using the amount of text per post and amount of karma as predictions

    plt.figure(figsize= (12,12))
    plt.plot(text_amount, predictions)
    sns.regplot(x= text_amount, y= predictions)
    plt.grid(True)
    plt.xlabel('Amount of Text Per Post')
    plt.ylabel('Upvotes')
    plt.title('Likes progression with ' + str(karmaCount) + ' karma')
    plt.show()

#Prediction based on 2000 author karma
PredictionsUsingKarma(gbr, 2000, stdsc)

#Prediction based on 10000 author karma
PredictionsUsingKarma(gbr, 10000, stdsc)

#Prediction based on 50000 author karma
PredictionsUsingKarma(gbr, 50000, stdsc)