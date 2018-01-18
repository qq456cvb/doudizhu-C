//
//  main.cpp
//  DouDiZhu
//
//  Created by Neil on 07/07/2017.
//  Copyright © 2017 Neil. All rights reserved.
//

#include <iostream>
#include <memory>
#include <algorithm>
#include "game.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::minus<T>());
    return result;
}

auto vector2numpy(const vector<int>& v) {
    if (v.size() == 0) return py::array_t<int>();
    auto result = py::array_t<int>(v.size());
    auto buf = result.request();
    int *ptr = (int*)buf.ptr;
    
    for (int i = 0; i < v.size(); ++i)
    {
        ptr[i] = v[i];
    }
    return result;
}

// 所有带2的方法都是双人争上游，不带2的是三人斗地主
class Env
{
public:
    Env() {
        this->reset();
    };
    Env(const Env& e) {
        indexID = e.indexID;
        clsGameSituation.reset(new GameSituation());
        *clsGameSituation = *e.clsGameSituation;
        uctALLCardsList.reset(new ALLCardsList());
        *uctALLCardsList = *e.uctALLCardsList;

        for(int i = 0; i < 3; i++) {
            arrHandCardData[i] = e.arrHandCardData[i];
        }
        value_lastCards = e.value_lastCards;
    }
    ~Env() {};
    
    std::shared_ptr<GameSituation> clsGameSituation;
    std::shared_ptr<ALLCardsList>  uctALLCardsList;

    int indexID = -1;

    HandCardData arrHandCardData[3];
    std::vector<int> value_lastCards;
    int last_category_idx = -1;

    void reset() {
        clsGameSituation.reset(new GameSituation());
        uctALLCardsList.reset(new ALLCardsList());
        memset(arrHandCardData, 0, sizeof(HandCardData) * 3);
        value_lastCards.clear();
    }

    auto getCurrID() {
        return indexID;
    }

    // 2 ：地主， 1： 地主上家，3：地主下家
    auto getRoleID() {
        if (indexID == clsGameSituation->nDiZhuID) return 2;
        if (indexID == clsGameSituation->nDiZhuID - 1 || indexID == clsGameSituation->nDiZhuID + 2) return 1;
        if (indexID == clsGameSituation->nDiZhuID + 1 || indexID == clsGameSituation->nDiZhuID - 2) return 3;
        return -1;
    }

    // one hot presentation: self_cards, remain_cards, [history 0-1], [0...]
    // 54 * 6 dimension (54 * 4 + 54 * 2 [0] paddings)
    py::array_t<int> getState2() {
        auto self_cards = toOneHot(arrHandCardData[indexID].color_nHandCardList);

        std::vector<int> state = self_cards;
        state += toOneHot(arrHandCardData[1-indexID].color_nHandCardList);
        state += toOneHot(clsGameSituation->color_aUnitOutCardList[0]);
        state += toOneHot(clsGameSituation->color_aUnitOutCardList[1]);
        state += vector<int>(54*2, 0);

        auto result = py::array_t<int>(state.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;
        
        for (int i = 0; i < state.size(); ++i)
        {
            ptr[i] = state[i];
        }
        return result;
    }

    // 静态方法，获取当前手牌的估值，输入:[0-56 color cards]
    static auto get_cards_value(py::array_t<int> pycards = py::array_t<int>()) {
        auto c = pycards.unchecked<1>();
        vector<int> cards;
        for (int i = 0; i < c.shape(0); i++) {
            cards.push_back(c[i]);
        }
        HandCardData data;
        data.color_nHandCardList = cards;
        data.Init();
        auto value = get_HandCardValue(data);
        return std::make_tuple(value.SumValue, value.NeedRound);
    }

    // 随机发牌
    void prepare2(py::array_t<int> pycards = py::array_t<int>()) {
        SendCards2(pycards, *clsGameSituation, *uctALLCardsList);

        arrHandCardData[0].color_nHandCardList = (*uctALLCardsList).arrCardsList[0];
        arrHandCardData[1].color_nHandCardList = (*uctALLCardsList).arrCardsList[1];

        for (int i = 0; i < 2; i++)
        {
            arrHandCardData[i].Init();
            arrHandCardData[i].nOwnIndex = i;
        }
        // py::print("0 player cards:");
        // arrHandCardData[0].PrintAll();
        // py::print("1 player cards:");
        // arrHandCardData[1].PrintAll();

        indexID = 0;
        clsGameSituation->nCardDroit = 0;
    }

    // 指定发牌
    void prepare2_manual(py::array_t<int> pycards) {
        SendCards2_manual(pycards, *clsGameSituation, *uctALLCardsList);

        arrHandCardData[0].color_nHandCardList = (*uctALLCardsList).arrCardsList[0];
        arrHandCardData[1].color_nHandCardList = (*uctALLCardsList).arrCardsList[1];

        for (int i = 0; i < 2; i++)
        {
            arrHandCardData[i].Init();
            arrHandCardData[i].nOwnIndex = i;
        }
        // py::print("0 player cards:");
        // arrHandCardData[0].PrintAll();
        // py::print("1 player cards:");
        // arrHandCardData[1].PrintAll();

        indexID = 0;
        clsGameSituation->nCardDroit = 0;
    }

    auto step2_auto() {
        get_PutCardList_2(*clsGameSituation, arrHandCardData[indexID]);
        arrHandCardData[indexID].PutCards();
        auto put_list = arrHandCardData[indexID].value_nPutCardList;
        std::sort(put_list.begin(), put_list.end());
        // printf("type: %d\n", (int)arrHandCardData[indexID].uctPutCardType.cgType);
        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;

        if (arrHandCardData[indexID].nHandCardCount == 0)
        {
            clsGameSituation->Over = true;
            return std::make_tuple(vector2numpy(put_list), true);
        }
        
        if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
        {
            clsGameSituation->nCardDroit = indexID;
            clsGameSituation->uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
            value_lastCards = arrHandCardData[indexID].value_nPutCardList;
        }
        indexID == 1 ? indexID = 0 : indexID++;

        return std::make_tuple(vector2numpy(put_list), false);
    }

    // 进行一步，若下面是电脑继续一步，输入3-17 value cards
    std::tuple<int,bool> step2(py::array_t<int> cards = py::array_t<int>()) {
        if (indexID == 1)
        {
            get_PutCardList_2(*clsGameSituation, arrHandCardData[indexID]);
        } else {
            arrHandCardData[indexID].ClearPutCardList();
            if (cards.size() == 0)
            {
                arrHandCardData[indexID].uctPutCardType = get_GroupData(cgZERO, 0, 0);
            } else {
                auto arr = cards.unchecked<1>();
                int cnt[18] = { 0 };
                for (int i = 0; i < arr.shape(0); ++i)
                {
                    arrHandCardData[indexID].value_nPutCardList.push_back(arr[i]);
                    cnt[arr[i]]++;
                }
                arrHandCardData[indexID].uctPutCardType = ins_SurCardsType(cnt);
            }
        }
        arrHandCardData[indexID].PutCards();
        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;


        // const auto &intention = arrHandCardData[indexID].value_nPutCardList;
        // py::print(indexID, " gives cards:", "sep"_a="");
        // for (vector<int>::iterator iter = arrHandCardData[indexID].color_nPutCardList.begin();
        //         iter != arrHandCardData[indexID].color_nPutCardList.end(); iter++)
        //     py::print(get_CardsName(*iter), "end"_a=(iter == arrHandCardData[indexID].color_nPutCardList.end() - 1 ? '\n' : ','));
        // py::print("");
        
        
        
        if (arrHandCardData[indexID].nHandCardCount == 0)
        {
            clsGameSituation->Over = true;
            
            if (indexID == 0)
            {
                // py::print("player ", indexID, " wins", "sep"_a="");
                return std::make_tuple(1, true);
            }
            else
            {
                // py::print("player ", indexID, " wins", "sep"_a="");
                return std::make_tuple(-1, true);
            }
        }
        
        if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
        {
            clsGameSituation->nCardDroit = indexID;
            clsGameSituation->uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
            value_lastCards = arrHandCardData[indexID].value_nPutCardList;
        }
        indexID == 1 ? indexID = 0 : indexID++;

        if (indexID == 1)
        {
            return step2();
        }
        return std::make_tuple(0, false);
    }

    void prepare() {

        SendCards(*clsGameSituation, *uctALLCardsList);
        
        arrHandCardData[0].color_nHandCardList = (*uctALLCardsList).arrCardsList[0];
        arrHandCardData[1].color_nHandCardList = (*uctALLCardsList).arrCardsList[1];
        arrHandCardData[2].color_nHandCardList = (*uctALLCardsList).arrCardsList[2];
        
        for (int i = 0; i < 3; i++)
        {
            arrHandCardData[i].Init();
            arrHandCardData[i].nOwnIndex = i;
        }

        // 如果无法打印，改成英文或者去掉py::print
        // py::print("0号玩家牌为：");
        // arrHandCardData[0].PrintAll();
        // py::print("1号玩家牌为：");
        // arrHandCardData[1].PrintAll();
        // py::print("2号玩家牌为：");
        // arrHandCardData[2].PrintAll();
        
        // py::print("底牌为：");
        // py::print(get_CardsName((*clsGameSituation).DiPai[0]), 
        //     get_CardsName((*clsGameSituation).DiPai[1]),
        //     get_CardsName((*clsGameSituation).DiPai[2]), "sep"_a=",");


        // call for lord
        for (int i = 0; i < 3; i++)
        {
            int  tmpLandScore = LandScore(*clsGameSituation, arrHandCardData[i]);
            if (tmpLandScore > clsGameSituation->nNowLandScore)
            {
                clsGameSituation->nNowLandScore = tmpLandScore;
                clsGameSituation->nNowDiZhuID = i;
            }
        }
        
        if (clsGameSituation->nNowDiZhuID == -1)
        {
            // py::print("No one calls for 地主");
            reset();
            return prepare();
        }
        
        // py::print(clsGameSituation->nNowDiZhuID, "号是地主，分为：", clsGameSituation->nNowLandScore, "sep"_a="");
        clsGameSituation->nDiZhuID=clsGameSituation->nNowDiZhuID;
        clsGameSituation->nLandScore =clsGameSituation->nNowLandScore;
        
        
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[0]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[1]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[2]);
        
        arrHandCardData[clsGameSituation->nDiZhuID].Init();
        
        indexID = clsGameSituation->nDiZhuID;
        
        // py::print();
        
        
        // py::print("0号玩家牌为：");
        // arrHandCardData[0].PrintAll();
        // py::print("1号玩家牌为：");
        // arrHandCardData[1].PrintAll();
        // py::print("2号玩家牌为：");
        // arrHandCardData[2].PrintAll();  

        clsGameSituation->nCardDroit = indexID;
    }

    int prepare_manual(py::array_t<int> pycards) {
        SendCards_manual(pycards, *clsGameSituation, *uctALLCardsList);
        
        arrHandCardData[0].color_nHandCardList = (*uctALLCardsList).arrCardsList[0];
        arrHandCardData[1].color_nHandCardList = (*uctALLCardsList).arrCardsList[1];
        arrHandCardData[2].color_nHandCardList = (*uctALLCardsList).arrCardsList[2];
        
        for (int i = 0; i < 3; i++)
        {
            arrHandCardData[i].Init();
            arrHandCardData[i].nOwnIndex = i;
        }


        // call for lord
        for (int i = 0; i < 3; i++)
        {
            int  tmpLandScore = LandScore(*clsGameSituation, arrHandCardData[i]);
            if (tmpLandScore > clsGameSituation->nNowLandScore)
            {
                clsGameSituation->nNowLandScore = tmpLandScore;
                clsGameSituation->nNowDiZhuID = i;
            }
        }
        
        if (clsGameSituation->nNowDiZhuID == -1)
        {
            reset();
            return prepare_manual(pycards);
        }
        
        clsGameSituation->nDiZhuID=clsGameSituation->nNowDiZhuID;
        clsGameSituation->nLandScore =clsGameSituation->nNowLandScore;
        
        
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[0]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[1]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[2]);
        
        arrHandCardData[clsGameSituation->nDiZhuID].Init();
        
        indexID = clsGameSituation->nDiZhuID;

        clsGameSituation->nCardDroit = indexID;
        return indexID;
    }

    // 转one hot, 输入[0-53 color cards]
    std::vector<int> toOneHot(const std::vector<int>& v) {
        std::vector<int> result(54, 0);
        for (auto color : v) {
            if (color > 52)
            {
                color = 53;
            }
            result[color]++;
        }
        return result;
    }

    // one hot presentation: self_cards, remain_cards, [history 0-2], extra_cards
    // 54 * 6 dimension
    py::array_t<int> getState() {
        std::vector<int> state;
        std::vector<int> total(54, 1);
        auto self_cards = toOneHot(arrHandCardData[indexID].color_nHandCardList);

        std::vector<int> remains = total - self_cards;
        vector<int> history[3] = {
            toOneHot(clsGameSituation->color_aUnitOutCardList[0]),
            toOneHot(clsGameSituation->color_aUnitOutCardList[1]),
            toOneHot(clsGameSituation->color_aUnitOutCardList[2])
        };
        for (int i = 0; i < 3; i++) {
            remains = remains - history[i];
        }

        vector<int> extra_cards(std::begin(clsGameSituation->DiPai), std::end(clsGameSituation->DiPai));
        extra_cards = toOneHot(extra_cards);
        state += self_cards;
        state += remains;
        state += history[0];
        state += history[1];
        state += history[2];
        state += extra_cards;


        auto result = py::array_t<int>(state.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;
        
        for (int i = 0; i < state.size(); ++i)
        {
            ptr[i] = state[i];
        }
        return result;
    }

    // 0-14 presentation
    // py::array_t<int> getState() {
    //     std::vector<int> state;
    //     std::vector<int> total(54, 1);
    //     auto self_cards = toOneHot(arrHandCardData[indexID].color_nHandCardList);

    //     std::vector<int> remains = total - self_cards;
    //     vector<int> history[3] = {
    //         toOneHot(clsGameSituation->color_aUnitOutCardList[0]),
    //         toOneHot(clsGameSituation->color_aUnitOutCardList[1]),
    //         toOneHot(clsGameSituation->color_aUnitOutCardList[2])
    //     };
    //     for (int i = 0; i < 3; i++) {
    //         remains = remains - history[i];
    //     }

    //     vector<int> extra_cards(std::begin(clsGameSituation->DiPai), std::end(clsGameSituation->DiPai));
    //     extra_cards = toOneHot(extra_cards);
    //     for (int i = 0; i < 54; i++) {
    //         if (remains[i] > 0) {
    //             if (i == 53) state.push_back(14);
    //             else state.push_back(i / 4);
    //         }
    //     }
    //     for (int i = 0; i < 54; i++) {
    //         if (self_cards[i] > 0) {
    //             if (i == 53) state.push_back(14);
    //             else state.push_back(i / 4);
    //         }
    //     }
    //     for (int j = 0; j < 3; j++) {
    //         for (int i = 0; i < 54; i++) {
    //             if (history[j][i] > 0) {
    //                 if (i == 53) state.push_back(14);
    //                 else state.push_back(i / 4);
    //             }
    //         }
    //     }
    //     for (int i = 0; i < 54; i++) {
    //         if (extra_cards[i] > 0) {
    //             if (i == 53) state.push_back(14);
    //             else state.push_back(i / 4);
    //         }
    //     }

    //     auto result = py::array_t<int>(state.size());
    //     auto buf = result.request();
    //     int *ptr = (int*)buf.ptr;
        
    //     for (int i = 0; i < state.size(); ++i)
    //     {
    //         ptr[i] = state[i];
    //     }
    //     return result;
    // }

    // (0-56 color cards 转 3-17 value cards)
    vector<int> toValue(const vector<int>& cards) {
        vector<int> result(cards.size(), 0);
        for (int i = 0; i < cards.size(); i++) {
            result[i] = cards[i] / 4 + 3;
        }
        return result;
    }

    // 获得地主手中的牌数
    int getLordCnt() {
        return (int)arrHandCardData[clsGameSituation->nDiZhuID].value_nHandCardList.size();
    }

    // 获得当前玩家手中的牌 3-17 value cards
    py::array_t<int> getCurrCards() {
        auto self_cards = arrHandCardData[indexID].value_nHandCardList;

        auto result = py::array_t<int>(self_cards.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;
        
        for (int i = 0; i < self_cards.size(); ++i)
        {
            ptr[i] = self_cards[i];
        }
        return result;
    }

    py::array_t<int> getCurrValueCards() {
        auto self_cards = arrHandCardData[indexID].color_nHandCardList;

        auto result = py::array_t<int>(self_cards.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;
        
        for (int i = 0; i < self_cards.size(); ++i)
        {
            ptr[i] = self_cards[i];
        }
        return result;
    }

    // 获得上家出的牌，如果自己控手，返回空
    py::array_t<int> getLastCards() {
        if (clsGameSituation->nCardDroit == indexID)
        {
            return py::array_t<int>();
        }
        auto result = py::array_t<int>(value_lastCards.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;
        
        for (int i = 0; i < value_lastCards.size(); ++i)
        {
            ptr[i] = value_lastCards[i];
        }
        return result;
    }

    // 获得上家出的牌的类型，如果自己控手，返回-1
    int getLastCategory() {
        if (clsGameSituation->nCardDroit == indexID)
        {
            return -1;
        }
        return last_category_idx;
    }

     auto step(bool lord = false, py::array_t<int> cards = py::array_t<int>()) {
        if (lord)
        {
            get_PutCardList_2(*clsGameSituation, arrHandCardData[indexID]);
        } else {
            arrHandCardData[indexID].ClearPutCardList();
            if (cards.size() == 0)
            {
                arrHandCardData[indexID].uctPutCardType = get_GroupData(cgZERO, 0, 0);
            } else {
                auto arr = cards.unchecked<1>();
                int cnt[18] = { 0 };
                for (int i = 0; i < arr.shape(0); ++i)
                {
                    arrHandCardData[indexID].value_nPutCardList.push_back(arr[i]);
                    cnt[arr[i]]++;
                }
                arrHandCardData[indexID].uctPutCardType = ins_SurCardsType(cnt);
            }
        }
        arrHandCardData[indexID].PutCards();
        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;

        // get group category
        auto category = arrHandCardData[indexID].uctPutCardType.cgType;
        int category_idx = 0;
        switch(category) {
            case cgZERO:
                category_idx = 0;
                break;
            case cgSINGLE:
                category_idx = 1;
                break;
            case cgDOUBLE:
                category_idx = 2;
                break;
            case cgTHREE:
                category_idx = 3;
                break;
            case cgBOMB_CARD:
                category_idx = 4;
                break;
            case cgTHREE_TAKE_ONE:
                category_idx = 5;
                break;
            case cgTHREE_TAKE_TWO:
                category_idx = 6;
                break;
            case cgSINGLE_LINE:
                category_idx = 7;
                break;
            case cgDOUBLE_LINE:
                category_idx = 8;
                break;
            case cgTHREE_LINE:
                category_idx = 9;
                break;
            case cgTHREE_TAKE_ONE_LINE:
                category_idx = 10;
                break;
            case cgTHREE_TAKE_TWO_LINE:
                category_idx = 11;
                break;
            case cgKING_CARD:
                category_idx = 12;
                break;
            case cgFOUR_TAKE_ONE:
                category_idx = 13;
                break;
            case cgFOUR_TAKE_TWO:
                category_idx = 14;
                break;
        }

        const auto &intention = arrHandCardData[indexID].value_nPutCardList;
        // check for bomb
        bool bomb = false;
        if (intention.size() == 4)
        {
            bomb = true;
            for (int i = 1; i < 4; ++i)
            {
                if (intention[i] != intention[0])
                {
                    bomb = false;
                    break;
                }
            }
        } else if (intention.size() == 2)
        {
            if ((intention[0] == 16 && intention[1] == 17) || (intention[0] == 17 && intention[1] == 16))
            {
                bomb = true;
            }
        }

        if (bomb)
        {
            clsGameSituation->nMultiple *= 2;
        }
        // py::print(indexID, "号玩家出牌：", "sep"_a="");
        // for (vector<int>::iterator iter = arrHandCardData[indexID].color_nPutCardList.begin();
        //         iter != arrHandCardData[indexID].color_nPutCardList.end(); iter++)
        //     py::print(get_CardsName(*iter), "end"_a=(iter == arrHandCardData[indexID].color_nPutCardList.end() - 1 ? '\n' : ','));
        // py::print("");
        
        
        
        if (arrHandCardData[indexID].nHandCardCount == 0)
        {
            clsGameSituation->Over = true;
            
            if (indexID == clsGameSituation->nDiZhuID)
            {
                // py::print("地主 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(-clsGameSituation->nLandScore * clsGameSituation->nMultiple, true, category_idx);
            }
            else
            {
                // py::print("农民 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(clsGameSituation->nLandScore * clsGameSituation->nMultiple, true, category_idx);
            }
        }
        
        if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
        {
            clsGameSituation->nCardDroit = indexID;
            clsGameSituation->uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
            value_lastCards = arrHandCardData[indexID].value_nPutCardList;
            last_category_idx = category_idx;
        }
        indexID == 2 ? indexID = 0 : indexID++;

        if (indexID == clsGameSituation->nDiZhuID)
        {
            return step(true);
        }
        return std::make_tuple(0, false, category_idx);
    }

    auto step_manual(py::array_t<int> cards = py::array_t<int>()) {
        arrHandCardData[indexID].ClearPutCardList();
        if (cards.size() == 0)
        {
            arrHandCardData[indexID].uctPutCardType = get_GroupData(cgZERO, 0, 0);
        } else {
            auto arr = cards.unchecked<1>();
            int cnt[18] = { 0 };
            for (int i = 0; i < arr.shape(0); ++i)
            {
                arrHandCardData[indexID].value_nPutCardList.push_back(arr[i]);
                cnt[arr[i]]++;
            }
            arrHandCardData[indexID].uctPutCardType = ins_SurCardsType(cnt);
        }
        arrHandCardData[indexID].PutCards();
        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;

        // get group category
        auto category = arrHandCardData[indexID].uctPutCardType.cgType;
        int category_idx = 0;
        switch(category) {
            case cgZERO:
                category_idx = 0;
                break;
            case cgSINGLE:
                category_idx = 1;
                break;
            case cgDOUBLE:
                category_idx = 2;
                break;
            case cgTHREE:
                category_idx = 3;
                break;
            case cgBOMB_CARD:
                category_idx = 4;
                break;
            case cgTHREE_TAKE_ONE:
                category_idx = 5;
                break;
            case cgTHREE_TAKE_TWO:
                category_idx = 6;
                break;
            case cgSINGLE_LINE:
                category_idx = 7;
                break;
            case cgDOUBLE_LINE:
                category_idx = 8;
                break;
            case cgTHREE_LINE:
                category_idx = 9;
                break;
            case cgTHREE_TAKE_ONE_LINE:
                category_idx = 10;
                break;
            case cgTHREE_TAKE_TWO_LINE:
                category_idx = 11;
                break;
            case cgKING_CARD:
                category_idx = 12;
                break;
            case cgFOUR_TAKE_ONE:
                category_idx = 13;
                break;
            case cgFOUR_TAKE_TWO:
                category_idx = 14;
                break;
        }

        const auto &intention = arrHandCardData[indexID].value_nPutCardList;
        // check for bomb
        bool bomb = false;
        if (intention.size() == 4)
        {
            bomb = true;
            for (int i = 1; i < 4; ++i)
            {
                if (intention[i] != intention[0])
                {
                    bomb = false;
                    break;
                }
            }
        } else if (intention.size() == 2)
        {
            if ((intention[0] == 16 && intention[1] == 17) || (intention[0] == 17 && intention[1] == 16))
            {
                bomb = true;
            }
        }

        if (bomb)
        {
            clsGameSituation->nMultiple *= 2;
        }
        // py::print(indexID, "号玩家出牌：", "sep"_a="");
        // for (vector<int>::iterator iter = arrHandCardData[indexID].color_nPutCardList.begin();
        //         iter != arrHandCardData[indexID].color_nPutCardList.end(); iter++)
        //     py::print(get_CardsName(*iter), "end"_a=(iter == arrHandCardData[indexID].color_nPutCardList.end() - 1 ? '\n' : ','));
        // py::print("");
        
        
        
        if (arrHandCardData[indexID].nHandCardCount == 0)
        {
            clsGameSituation->Over = true;
            
            if (indexID == clsGameSituation->nDiZhuID)
            {
                // py::print("地主 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(-clsGameSituation->nLandScore * clsGameSituation->nMultiple, true, category_idx);
            }
            else
            {
                // py::print("农民 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(clsGameSituation->nLandScore * clsGameSituation->nMultiple, true, category_idx);
            }
        }
        
        if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
        {
            clsGameSituation->nCardDroit = indexID;
            clsGameSituation->uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
            value_lastCards = arrHandCardData[indexID].value_nPutCardList;
            last_category_idx = category_idx;
        }
        indexID == 2 ? indexID = 0 : indexID++;

        return std::make_tuple(0, false, category_idx);
    }

    // 3-18 value cards
    auto will_lose_control(py::array_t<int> cards) {
        auto bkup = clsGameSituation->uctNowCardGroup;
        int cnt[18] = { 0 };
        auto c = cards.unchecked<1>();
        for (int i = 0; i < c.shape(0); i++) {
            cnt[c[i]]++;
        }
        clsGameSituation->uctNowCardGroup = ins_SurCardsType(cnt);
        
        get_PutCardList_2(*clsGameSituation, arrHandCardData[(indexID + 1) % 3]);
        clsGameSituation->uctNowCardGroup = bkup;
        if (arrHandCardData[(indexID + 1) % 3].uctPutCardType.cgType != cgZERO) {
            return true;
        } else {
            return false;
        }
    }


    auto step_trial(bool lord = false, py::array_t<int> cards = py::array_t<int>()) {
        // printf("current id: %d\n", indexID);
        Env env(*this);
        // printf("temp id: %d\n", env.indexID);
        auto tuple = env.step(lord, cards);
        // notice we are making some states map only to values but not actions,
        // e.g. agent1 step-> agent2's state, but agent1 cannot take action when agnet2 is in control.
        return std::make_tuple(std::get<0>(tuple), std::get<0>(Env::get_cards_value(env.getCurrValueCards())), env.getState());
    }

   bool is_bomb(std::vector<int> vals) {
       if (vals.size() != 4) return false;
       if (vals[0] != vals[3]) return false;
       return true;
   }

    // handcards: 0-56 color cards, last_cards : 3 - 18 value cards
    static auto step_auto_static(py::array_t<int> handcards, py::array_t<int> last_cards = py::array_t<int>()) {
        auto c = handcards.unchecked<1>();
        vector<int> cards;
        for (int i = 0; i < c.shape(0); i++) {
            cards.push_back(c[i]);
        }
        HandCardData data;
        data.color_nHandCardList = cards;
        data.Init();

        if (last_cards.size() == 0) {
            get_PutCardList_2_unlimit(data);
        } else {
            cards.clear();
            auto last_cards_ptr = last_cards.unchecked<1>();
            for (int i = 0; i < last_cards_ptr.shape(0); i++) {
                cards.push_back(last_cards_ptr[i]);
            }
            GameSituation sit;
            sit.uctNowCardGroup = ins_SurCardsType(cards);
            get_PutCardList_2_limit(sit, data);
        }
        auto intention = data.value_nPutCardList;
        return vector2numpy(intention);
    }

    auto step_auto() {
        get_PutCardList_2(*clsGameSituation, arrHandCardData[indexID]);
        arrHandCardData[indexID].PutCards();
        auto intention = arrHandCardData[indexID].value_nPutCardList;
        
        //std::sort(intention.begin(), intention.end());

        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;
        
        // get group category
        auto category = arrHandCardData[indexID].uctPutCardType.cgType;
        int category_idx = 0;
        switch(category) {
            case cgZERO:
                category_idx = 0;
                break;
            case cgSINGLE:
                category_idx = 1;
                break;
            case cgDOUBLE:
                category_idx = 2;
                break;
            case cgTHREE:
                category_idx = 3;
                break;
            case cgBOMB_CARD:
                category_idx = 4;
                break;
            case cgTHREE_TAKE_ONE:
                category_idx = 5;
                break;
            case cgTHREE_TAKE_TWO:
                category_idx = 6;
                break;
            case cgSINGLE_LINE:
                category_idx = 7;
                break;
            case cgDOUBLE_LINE:
                category_idx = 8;
                break;
            case cgTHREE_LINE:
                category_idx = 9;
                break;
            case cgTHREE_TAKE_ONE_LINE:
                category_idx = 10;
                break;
            case cgTHREE_TAKE_TWO_LINE:
                category_idx = 11;
                break;
            case cgKING_CARD:
                category_idx = 12;
                break;
            case cgFOUR_TAKE_ONE:
                category_idx = 13;
                break;
            case cgFOUR_TAKE_TWO:
                category_idx = 14;
                break;
        }
        // check for bomb
        bool bomb = false;
        if (intention.size() == 4) {
            if (intention[0] == intention[3]) bomb = true;
        } else if (intention.size() == 2) {
            if (intention[0] == 16 && intention[1] == 17) bomb = true;
        }

        if (bomb)
        {
            clsGameSituation->nMultiple *= 2;
        }
        
        
        
        if (arrHandCardData[indexID].nHandCardCount == 0)
        {
            clsGameSituation->Over = true;
            
            if (indexID == clsGameSituation->nDiZhuID)
            {
                return std::make_tuple(vector2numpy(intention), -clsGameSituation->nLandScore * clsGameSituation->nMultiple, category_idx);
            }
            else
            {
                return std::make_tuple(vector2numpy(intention), clsGameSituation->nLandScore * clsGameSituation->nMultiple, category_idx);
            }
        }
        
        if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
        {
            clsGameSituation->nCardDroit = indexID;
            clsGameSituation->uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
            value_lastCards = arrHandCardData[indexID].value_nPutCardList;
            last_category_idx = category_idx;
        }
        indexID == 2 ? indexID = 0 : indexID++;

        return std::make_tuple(vector2numpy(intention), 0, category_idx);
    }
};

PYBIND11_MODULE(env, m) {
    py::class_<Env>(m, "Env")
        .def(py::init<>())
        .def("reset", &Env::reset)
        .def("prepare", &Env::prepare)
        .def("prepare_manual", &Env::prepare_manual)
        .def("prepare2", &Env::prepare2)
        .def("prepare2_manual", &Env::prepare2_manual)
        .def("get_state", &Env::getState)
        .def("get_state2", &Env::getState2)
        .def("step", &Env::step, py::arg("lord") = false, py::arg("cards") = py::array_t<int>())
        .def("step_manual", &Env::step_manual, py::arg("cards") = py::array_t<int>())
        .def("step_trial", &Env::step_trial, py::arg("lord") = false, py::arg("cards") = py::array_t<int>())
        .def("step_auto", &Env::step_auto)
        .def_static("step_auto_static", &Env::step_auto_static)
        .def("step2", &Env::step2, py::arg("cards") = py::array_t<int>())
        .def("step2_auto", &Env::step2_auto)
        .def("will_lose_control", &Env::will_lose_control)
        .def_static("get_cards_value", &Env::get_cards_value)
        .def("get_curr_ID", &Env::getCurrID)
        .def("get_role_ID", &Env::getRoleID)
        .def("get_curr_handcards", &Env::getCurrCards)
        .def("get_last_outcards", &Env::getLastCards)
        .def("get_last_outcategory_idx", &Env::getLastCategory)
        .def("get_lord_cnt", &Env::getLordCnt);
}

// int main(int argc, const char * argv[]) {
//     GameSituation clsGameSituation;
    
//     ALLCardsList  uctALLCardsList;
    
//     //发牌
//     SendCards(clsGameSituation, uctALLCardsList);
    
//     HandCardData arrHandCardData[3];
    
//     arrHandCardData[0].color_nHandCardList = uctALLCardsList.arrCardsList[0];
//     arrHandCardData[1].color_nHandCardList = uctALLCardsList.arrCardsList[1];
//     arrHandCardData[2].color_nHandCardList = uctALLCardsList.arrCardsList[2];
    
//     for (int i = 0; i < 3; i++)
//     {
//         arrHandCardData[i].Init();
//         arrHandCardData[i].nOwnIndex = i;
//     }
    
//     cout << "0号玩家牌为：" << endl;
//     arrHandCardData[0].PrintAll();
//     cout << "1号玩家牌为：" << endl;
//     arrHandCardData[1].PrintAll();
//     cout << "2号玩家牌为：" << endl;
//     arrHandCardData[2].PrintAll();
    
//     cout << "底牌为：" << endl;
//     cout << get_CardsName(clsGameSituation.DiPai[0]) << ','
//     << get_CardsName(clsGameSituation.DiPai[1]) << ','
//     << get_CardsName(clsGameSituation.DiPai[2]) << endl;  
    
//     cout << endl;
    
//     for (int i = 0; i < 3; i++)
//     {
//         int  tmpLandScore = LandScore(clsGameSituation, arrHandCardData[i]);
//         if (tmpLandScore > clsGameSituation.nNowLandScore)
//         {
//             clsGameSituation.nNowLandScore = tmpLandScore;
//             clsGameSituation.nNowDiZhuID = i;
//         }
//     }
    
//     if (clsGameSituation.nNowDiZhuID == -1)
//     {
//         cout << "无人叫地主" << endl;
//         return 0;
//     }
    
//     cout << clsGameSituation.nNowDiZhuID << "号玩家是地主，叫分为：" << clsGameSituation.nNowLandScore << endl;
//     clsGameSituation.nDiZhuID=clsGameSituation.nNowDiZhuID;
//     clsGameSituation.nLandScore =clsGameSituation.nNowLandScore;
    
    
//     //将三张底牌给地主
//     arrHandCardData[clsGameSituation.nDiZhuID].color_nHandCardList.push_back(clsGameSituation.DiPai[0]);
//     arrHandCardData[clsGameSituation.nDiZhuID].color_nHandCardList.push_back(clsGameSituation.DiPai[1]);
//     arrHandCardData[clsGameSituation.nDiZhuID].color_nHandCardList.push_back(clsGameSituation.DiPai[2]);
    
//     //地主手牌刷新
//     arrHandCardData[clsGameSituation.nDiZhuID].Init();
    
//     //出牌玩家ID
//     int indexID= clsGameSituation.nDiZhuID;
    
//     cout << endl;
    
    
//     cout << "0号玩家牌为：" << endl;
//     arrHandCardData[0].PrintAll();
//     cout << "1号玩家牌为：" << endl;
//     arrHandCardData[1].PrintAll();
//     cout << "2号玩家牌为：" << endl;  
//     arrHandCardData[2].PrintAll();  
//     //当前控手玩家先为地主  
//     clsGameSituation.nCardDroit = indexID;
    
//     while (!clsGameSituation.Over)
//     {
//         get_PutCardList_2(clsGameSituation, arrHandCardData[indexID]);//获取出牌序列
//         arrHandCardData[indexID].PutCards();
//         cout << indexID << "号玩家出牌：" << endl;
//         for (vector<int>::iterator iter = arrHandCardData[indexID].color_nPutCardList.begin();
//              iter != arrHandCardData[indexID].color_nPutCardList.end(); iter++)
//             cout << get_CardsName(*iter) << (iter == arrHandCardData[indexID].color_nPutCardList.end() - 1 ? '\n' : ',');
//         cout << endl;
        
//         if (arrHandCardData[indexID].nHandCardCount == 0)
//         {
//             clsGameSituation.Over = true;
            
//             if (indexID == clsGameSituation.nDiZhuID)
//             {
//                 cout << "地主" << indexID << "号玩家获胜" << endl;
//             }
//             else
//             {
//                 for (int i = 0; i < 3; i++) {
//                     if (i != clsGameSituation.nDiZhuID)
//                     {
//                         cout << "农民" << i << "号玩家获胜" << endl;
//                     }
//                 }
//             }
            
//         }
        
//         if (arrHandCardData[indexID].uctPutCardType.cgType != cgZERO)
//         {
//             clsGameSituation.nCardDroit = indexID;
//             clsGameSituation.uctNowCardGroup = arrHandCardData[indexID].uctPutCardType;
//         }
        
//         indexID == 2 ? indexID = 0 : indexID++;
        
//     }
// }
