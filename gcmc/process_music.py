# -*- coding: utf-8 -*-
import numpy as np
import json
import os.path

ESSENTIA_PATH = './drive/MyDrive/essentia/{}/{}/{}.mp3.json'

def get_sample_dict(all_features=True):
    # This is used to train DictVectorizer
    return get_audio_features_dict("00006c661b0c80ef519ba561e321d100", all_features)

def get_audio_features_dict(songid, all_features=True):
    # This method returns all the audio features of a song or only the highlevel features
    audio_path = ESSENTIA_PATH.format(songid[:2], songid[2:4], songid)
    audio_features_dict = None
    if os.path.isfile(audio_path):
        audio_features_dict = {}
        audio_features = json.load(open(audio_path), strict=False)
        #print("audio_features:", audio_features)
        features = {"lowlevel": ["average_loudness", "dissonance", "dynamic_complexity", "zerocrossingrate"],
                    "rhythm": ["bpm", "bpm_histogram_first_peak_bpm", "bpm_histogram_second_peak_bpm", "danceability" ,"onset_rate"],
                    "tonal": ["chords_changes_rate", "chords_number_rate", "chords_strength"]}
        if all_features:
            # Add mean and variance for all features in 'features'
            for k in features.keys():
                for f in features[k]:
                    if isinstance(audio_features[k][f], dict):
                        audio_features_dict["%s_var" % f] = audio_features[k][f]["var"]
                        audio_features_dict["%s_mean" % f] = audio_features[k][f]["mean"]
                    else:
                        audio_features_dict[f] = audio_features[k][f]

        # Add Tagtraum highlevel "class" features
        # We always add this features, it doesn't depend on 'all_features' parameter
        k = "tagtraum"
        for f in audio_features["highlevel"][k]["all"]:
            audio_features_dict["class_f_%s_%s" % (k,f)] = audio_features["highlevel"][k]["all"][f]

        if all_features:
            # Add MFCC and HPCP
            """
            for i,f in enumerate(audio_features["lowlevel"]["mfcc"]["mean"]):
                audio_features_dict["mfcc_%d"%i] = f
            for i,f in enumerate(audio_features["tonal"]["hpcp"]["var"]):
                audio_features_dict["hpcp_var_%d"%i] = f
            for i,f in enumerate(audio_features["tonal"]["hpcp"]["mean"]):
                audio_features_dict["hpcp_mean_%d"%i] = f
            """

            # Add key and scale
            audio_features_dict["class_f_key"] = audio_features["tonal"]["key_key"]
            audio_features_dict["class_f_scale"] = audio_features["tonal"]["key_scale"]
    return audio_features_dict

import pickle
import os
import re
import json
import numpy as np
import datetime
import sys

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
#import lightfm
#from lightfm import LightFM
from collections import defaultdict
#from audio_features import get_audio_features_dict
#from lightfm.evaluation import precision_at_k, auc_score

SEED = 10

#キャラクターの消去
def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def process_mpd(max_slice, max_challenge_slice):
    prev_songs_window = 10
    playlists_path = '../.././drive/MyDrive/mpd/data/data'
    target_playlists = '../.././drive/MyDrive/mpd/data/challenge_set.json'
    output_file = 'output_lightFM.csv'
    #max_slice = 1 #(1/10)本番消去 max=1000
    #max_challenge_slice = 10 #max=1000 

    max_prev_song = 0
    previous_tracks = defaultdict(lambda: defaultdict(int)) #d['key1']['key2']の定義
    playlists_tracks = []
    playlists = []
    playlists_extra = {'name': []}
    filenames = os.listdir(playlists_path)

    #playlist_pathのロード
    key_errors = 0
    time_slice = 0
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice") and filename.endswith(".json"):
            fullpath = os.sep.join((playlists_path, filename))
            #print(fullpath)
            f = open(fullpath, "r")
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            time_slice += 1
            if max_slice < time_slice:
                break
            for playlist in mpd_slice['playlists']:
                try:
                    nname = normalize_name(playlist['name'])
                except KeyError:
                    key_errors += 1
                playlists_extra['name'].append(nname)
                tracks = defaultdict(int)
                sorted_tracks = sorted(playlist['tracks'], key=lambda k: k['pos'])
                for track in sorted_tracks:
                    track_uri_key = track['track_uri']
                    tracks[track_uri_key] += 1
                playlists_tracks.append(tracks)
                playlists.append(str(playlist['pid']))
    if key_errors != 0:
        print("KeyErrors:" + str(key_errors))

    # Add playlists on testing set
    test_playlists = []
    target = json.load(open(target_playlists))
    train_playlists_count = len(playlists)
    test_playlists_recommended_sum = []
    print("len_playlists =", max_challenge_slice, str(len(target["playlists"])))
    
    i=0
    for playlist in target["playlists"]:
        i+=1
        if i > max_challenge_slice:
            break
        nname = ""
        if 'name' in playlist:
            nname = normalize_name(playlist['name'])
        playlists_extra['name'].append(nname)
        playlists.append(str(playlist['pid']))
        test_playlists.append(str(playlist['pid']))
        #0 trackのプレイリストにはtop人気のレコメンドをする
        if len(playlist['tracks']) == 0:
            playlists_tracks.append({})
            continue
        tracks = defaultdict(int)
        for track in playlist['tracks']:
            tracks[track['track_uri']] += 1
        playlists_tracks.append(tracks)

    print ("Data loaded. Creating features matrix")
    #playlist & tracks
    dv = DictVectorizer()
    interaction_matrix = dv.fit_transform(playlists_tracks)
    #playlist & name
    lb = LabelBinarizer(sparse_output=True)
    pfeat = lb.fit_transform(playlists_extra['name'])
    playlist_features = pfeat

    # Need to hstack(行列の連結) playlist_features
    eye = sparse.eye(playlist_features.shape[0], playlist_features.shape[0]).tocsr()
    playlist_features_concat = sparse.hstack((eye, playlist_features))

    #tracks & genres
    item_prev = []
    highlevel = []
    get_audio_time = 0
    len_dv_feature_names = len(dv.feature_names_)
    for track in dv.feature_names_:
        # try:
        #     f = get_audio_features_dict(track.replace('spotify:track:', ''), False)
        #     get_audio_time += 1
        #     if get_audio_time % 10000 == 0:
        #       print("get audio time :", get_audio_time, "/", len_dv_feature_names)
        # except ValueError:
        #     print("Failed loading json", track)
        #     f = None
        get_audio_time += 1
        #if get_audio_time % 10000 == 0:
          #print("get audio time :", get_audio_time, "/", len_dv_feature_names)
        f = None
        curr_highlevel = {}
        if f is not None:
            curr_highlevel = {k:v for k,v in f.items() if 'class_f' in k}
        highlevel.append(curr_highlevel)

    ifeat_highlevel = DictVectorizer().fit_transform(highlevel)
    item_prev = ifeat_highlevel
    eye = sparse.eye(item_prev.shape[0], item_prev.shape[0]).tocsr()
    item_feat = sparse.hstack((eye, item_prev))


    print("num_track", len_dv_feature_names)
    print("max_NUMCLASSES: ",interaction_matrix.max())
    print("item_feat_shape", item_feat.shape)
    print("user_feat_shape", playlist_features_concat.shape)
    print("Features matrix created.")

    return interaction_matrix, playlist_features_concat, item_feat, test_playlists, train_playlists_count, playlists_tracks 
