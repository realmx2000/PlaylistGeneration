# coding=utf-8
"""
Modified Tutorial for Million songs dataset used for data scraping
"""
import os
import sys
import time
import glob
import datetime
import sqlite3
import numpy as np 
import hdf5_getters as GETTERS

#Initialization
msd_subset_path='/Users/erikjones/Documents/Stanford 2017:18/Autumn CS229/Project/MillionSongSubset'
msd_subset_data_path=os.path.join(msd_subset_path,'data')
msd_subset_addf_path=os.path.join(msd_subset_path,'AdditionalFiles')
assert os.path.isdir(msd_subset_path),'wrong path'
msd_code_path='/Users/erikjones/Documents/Stanford 2017:18/Autumn CS229/Project/MSongsDB-master'
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
sys.path.append( os.path.join(msd_code_path,'PythonSrc') )

def construct_song_dict(song_list):
    """
    Given a list with elements "Artist: Song_title", returns a 
    dictionary mapping artists to lists of their songs."""
    song_dict = {}
    for elem in song_list:
        elem_list = elem.split(':')
        artist = elem_list[0].strip(' ')
        song = elem_list[1].strip(' ')
        if artist not in song_dict:
            song_dict[artist] = []
        song_dict[artist].append(song)
    return song_dict

def apply_to_all_files_mod(basedir, song_list, filename = 'songs.npy', func=lambda x: x,ext='.h5'):
    """
    From a base directory, goes through all subdirectories, finds all files with the given ext,
    and reads each song from each file. For each song in song_list, gets the title, artist, tempo, 
    familiarity, hottness, terms, dancebility, duration, energy, loudness, and the timbre matrix. 
    Tab delimits terms, flattens the timbre matrix, adds them all to a np array, and saves the array 
    with the information from each song to filename. 
    """
    #Initial list of desired song info
    csv_data = []
    count = 0
    done_gg = False
    song_dict = construct_song_dict(song_list)
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        #Iterates through each file in files
        for filename in files:
            count += 1
            if count % 1000 == 0:
                print count
            h5 = GETTERS.open_h5_file_read(filename)
            #Scrapes desired data
            title = GETTERS.get_title(h5)
            artist = GETTERS.get_artist_name(h5)
            tempo = GETTERS.get_tempo(h5)
            familiarity = GETTERS.get_artist_familiarity(h5)
            hotness = GETTERS.get_artist_hotttnesss(h5)
            terms = GETTERS.get_artist_terms(h5)
            danceability = GETTERS.get_danceability(h5)
            duration = GETTERS.get_duration(h5)
            energy = GETTERS.get_energy(h5)
            loudness = GETTERS.get_loudness(h5)
            timbre = GETTERS.get_segments_timbre(h5)
            #Tab delimits terms
            terms_tabs = "\t".join(terms)
            #Flattens timbre
            timbre_flattened = timbre.flatten()
            #Creates np array of everything but timbre matrix
            everything_but_timbre = np.array([title, artist, tempo, familiarity, hotness, terms_tabs, danceability, duration, energy, loudness])
            #Combines everything else with timbre matrix
            row = np.concatenate((everything_but_timbre, timbre_flattened))
            #Checks if artist, song combination was in the list and, if so, adds it.
            if artist in song_dict:
                if title in song_dict[artist]:
                    print("Adding {} by {}. Song ID is: {}".format(title, artist, GETTERS.get_song_id(h5)))
                    csv_data.append(row)
                    #Prevents duplicates
                    song_dict[artist][song_dict[artist].index(title)] = ''
            h5.close()


    print("Number of songs: {}, artists {}".format(len(csv_data), len(song_dict)))
    csv_array = np.array(csv_data)
    #Saves data
    np.save(filename, csv_array)
      

apply_to_all_files_mod(msd_subset_data_path, song_list = ["Kings Of Leon: Wicker Chair", "Foo Fighters: Hell", "Pearl Jam: Inside Job", "U2: Kite Live from Sydney", "The Notorious B.I.G.: Who Shot Ya (Amended Album Version)", "Led Zeppelin: Poor Tom (Album Version)", "Tom Petty: A Higher Place (Album Version)", "Oasis: Morning Glory", "SNOWPATROL: We Wish You A Merry Christmas", "George Harrison: Hear Me Lord (2001 Digital Remaster)", "The Police: When The World Is Running Down_ You Make The Best Of What's Still Around", "Kings Of Leon: Knocked Up", "Owl City: Panda Bear", "Shakira: Pienso En Ti", "Green Day: Wake Me Up When September Ends (Live at Foxboro_ MA 9/3/05)", "Simon & Garfunkel: Citizen Of The Planet", "Kelly Clarkson: My Life Would Suck Without You", "Christina Aguilera: Walk Away", "Tom Petty And The Heartbreakers: Mary Jane's Last Dance", "Snow Patrol: Half The Fun"])

