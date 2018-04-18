//
//  game.hpp
//  DouDiZhu
//
//  Created by Neil on 07/07/2017.
//  Copyright © 2017 Neil. All rights reserved.
//

#ifndef game_hpp
#define game_hpp

#include <stdio.h>
#include <vector>
#include "card.hpp"
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//AI

struct ALLCardsList
{
    std::vector<int>  arrCardsList[3];
};

class GameSituation
{
public:
    GameSituation()
    {
    }
    ~GameSituation()
    {
        for (int i = 0; i < 3; i++) color_aUnitOutCardList[i].clear();
    }
    
    
public:
    //
    int nDiZhuID = -1;
    //
    int nLandScore = 0;
    
    //
    int nNowDiZhuID = -1;
    //
    int nNowLandScore = 0;
    
    //
    int DiPai[3] = { 0 };
    //index0~4
    int value_aAllOutCardList[18] = { {0} };
    //
    int value_aUnitOutCardList[3][18] = { {0} };
    //
    int value_aUnitHandCardCount[3] = { 0 };
    //
    int nMultiple = 1;
    //
    int nCardDroit = 0;
    //
    CardGroupData uctNowCardGroup;
    //
    bool Over = false;
    
    std::vector<int> color_aUnitOutCardList[3];
};

int LandScore(GameSituation &clsGameSituation, HandCardData &clsHandCardData, int &);
void InitCards(vector <int> &Cards, std::mt19937& g);
void InitCards2(const py::array_t<int>& pycards, vector <int> &Cards);
void InitCards_Appoint(vector <int> &Cards);
void SendCards(GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList, std::mt19937& g);
void SendCards_manual(const py::array_t<int>& pycards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList, std::mt19937& g);
void SendCards2(const py::array_t<int>& init_cards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList);
void SendCards2_manual(const py::array_t<int>& pycards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList);

#endif /* game_hpp */
