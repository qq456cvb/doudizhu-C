//
//  game.cpp
//  DouDiZhu
//
//  Created by Neil on 07/07/2017.
//  Copyright © 2017 Neil. All rights reserved.
//

#include "game.hpp"
#include <random>

/*
 
 */

int LandScore(GameSituation &clsGameSituation, HandCardData &clsHandCardData, int &sum_value)
{
    
    clsHandCardData.uctHandCardValue=get_HandCardValue(clsHandCardData);
    
    sum_value = clsHandCardData.uctHandCardValue.SumValue;
    
//     cout << "SumValue is :" << sum_value << ",";
    
    // cout << "NeedRound is :" << clsHandCardData.uctHandCardValue.NeedRound << endl;
    
//    if (SumValue<10)
//    {
//        return 0;
//    }
//    else
    if (sum_value < 15)
    {
        return 1;
    }
    else if (sum_value < 20)
    {
        return 2;
    }
    else
    {
        return 3;
    }
}

//

void InitCards2(const py::array_t<int> &pycards, vector<int> &cards)
{
    auto init_cards = pycards.unchecked<1>();
    cards.clear();
    
    vector <int> tmpCards;
    int i;
    for (int i = 0; i < init_cards.shape(0); ++i) {
        tmpCards.push_back(init_cards[i]);
    }
    
    
    //
    for (i = tmpCards.size(); i>0; i--) {
        srand(unsigned(time(NULL)));
        // 
        int index = rand() % i;
        cards.push_back(tmpCards[index]);
        tmpCards.erase(tmpCards.begin() + index);
    }
}

void InitCards(vector <int> &Cards, std::mt19937& g)
{
    //Cards
//    Cards.clear();
//
//    vector <int> tmpCards;
//    int i;
//
//    //5652535455
    for (int i = 0; i < 53; i++) {
        Cards.push_back(i);
    }
    Cards.push_back(56);
    std::shuffle(Cards.begin(), Cards.end(), g);
//
//
//    //
//    for (i = tmpCards.size(); i>0; i--) {
//
//        //
//        int index = rand() % i;
//        Cards.push_back(tmpCards[index]);
//        tmpCards.erase(tmpCards.begin() + index);
//    }
    
}

//

void InitCards_Appoint(vector <int> &Cards)
{
    //Cards
    Cards.clear();
    
    /*********************/
    
    Cards.push_back(48); Cards.push_back(50); Cards.push_back(49);
    Cards.push_back(44); Cards.push_back(47); Cards.push_back(35);
    Cards.push_back(40); Cards.push_back(46); Cards.push_back(34);
    Cards.push_back(36); Cards.push_back(45); Cards.push_back(33);
    Cards.push_back(23); Cards.push_back(43); Cards.push_back(31);
    Cards.push_back(22); Cards.push_back(42);  Cards.push_back(30);
    Cards.push_back(21); Cards.push_back(41); Cards.push_back(29);
    Cards.push_back(19); Cards.push_back(39); Cards.push_back(27);
    Cards.push_back(18); Cards.push_back(38); Cards.push_back(26);
    Cards.push_back(17); Cards.push_back(37); Cards.push_back(25);
    Cards.push_back(15); Cards.push_back(32);  Cards.push_back(20);
    Cards.push_back(14); Cards.push_back(28); Cards.push_back(16);
    Cards.push_back(13); Cards.push_back(24); Cards.push_back(12);
    Cards.push_back(11); Cards.push_back(3); Cards.push_back(7);
    Cards.push_back(10); Cards.push_back(2); Cards.push_back(6);
    Cards.push_back(9); Cards.push_back(1); Cards.push_back(5);
    Cards.push_back(8); Cards.push_back(0); Cards.push_back(4);
    Cards.push_back(51); Cards.push_back(52); Cards.push_back(56);
    
    
}

//

void SendCards(GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList, std::mt19937& g)
{
    //
    vector <int> Cards;
    InitCards(Cards, g);
    //InitCards_Appoint(Cards);
    int i, j, k;
    j = 0;
    for (i = 0; i < 3; i++) {
        for (k = 0; k < 17; k++,j++) {
            uctALLCardsList.arrCardsList[i].push_back(Cards[j]);
        }
    }
    
    //
    clsGameSituation.DiPai[0] = Cards[j];
    clsGameSituation.DiPai[1] = Cards[j+1];
    clsGameSituation.DiPai[2] = Cards[j+2];
    
    return;
}

void SendCards_manual(const py::array_t<int>& pycards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList, std::mt19937& g)
{
    auto init_cards = pycards.unchecked<1>();
    vector<int> cards;
    int i;
    for (int i = 0; i < init_cards.shape(0); ++i) {
        cards.push_back(init_cards[i]);
    }
    std::shuffle(cards.begin(), cards.end(), g);

    int j, k;
    j = 0;
    for (i = 0; i < 3; i++) {
        for (k = 0; k < cards.size() / 3 - 1; k++, j++) {
            uctALLCardsList.arrCardsList[i].push_back(cards[j]);
            // std::cout << cards[j] << " ";
        }
    }
    // std::cout << std::endl;

    clsGameSituation.DiPai[0] = cards[j];
    clsGameSituation.DiPai[1] = cards[j+1];
    clsGameSituation.DiPai[2] = cards[j+2];
    
    return;
}

void SendCards2_manual(const py::array_t<int>& pycards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList) {
    auto init_cards = pycards.unchecked<1>();
    vector<int> cards;
    int i;
    for (int i = 0; i < init_cards.shape(0); ++i) {
        cards.push_back(init_cards[i]);
        // std::cout << init_cards[i] << " ";
    }
    // std::cout << std::endl;

    int j, k;
    j = 0;
    for (i = 0; i < 2; i++) {
        for (k = 0; k < cards.size() / 2; k++, j++) {
            uctALLCardsList.arrCardsList[i].push_back(cards[j]);
        }
    }
    
    return;
}

void SendCards2(const py::array_t<int>& init_cards, GameSituation & clsGameSituation, ALLCardsList &uctALLCardsList) {
    vector<int> cards;

    int i, j, k;
    j = 0;
    for (k = 0; k < cards.size() / 2; k++) {
        for (i = 0; i < 2; i++,j++)
        {
            uctALLCardsList.arrCardsList[i].push_back(cards[j]);
        }
    }
    HandCardData data1, data2;
    data1.color_nHandCardList = uctALLCardsList.arrCardsList[0];
    data2.color_nHandCardList = uctALLCardsList.arrCardsList[1];
    data1.Init();
    data2.Init();
    auto value1 = get_HandCardValue(data1);
    auto value2 = get_HandCardValue(data2);

    // 找一个相对比较好的局
    if (value1.SumValue < value2.SumValue + 5) {
        uctALLCardsList.arrCardsList[0].clear();
        uctALLCardsList.arrCardsList[1].clear();
        return SendCards2(init_cards, clsGameSituation, uctALLCardsList);
    }
    
    return;
}