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
#include <random>
#include "game.hpp"
#include "dancing_link.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
    Env(long long int seed = -1) {
        this->reset();
        if (seed == -1) {
            std::cout << "seeding with " << std::random_device{}() << std::endl;
            this->g = std::mt19937(std::random_device{}());
        }
        else {
            std::cout << "seeding with " << seed << std::endl;
            this->g = std::mt19937(seed);
        }
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
    std::mt19937 g;

    void reset() {
        clsGameSituation.reset(new GameSituation());
        uctALLCardsList.reset(new ALLCardsList());
        memset(arrHandCardData, 0, sizeof(HandCardData) * 3);
        value_lastCards.clear();
        last_category_idx = -1;
    }

    static void reorder_cards(vector<int>& cards, CardGroupType category) {
        switch(category) {
            case cgTHREE_TAKE_ONE: {
                if (cards[0] != cards[1]) {
                    std::swap(cards[0], cards[3]);
                }
                break;
            }
            case cgTHREE_TAKE_TWO: {
                int i;
                for (i = 0; i < 4; i++) {
                    if (cards[i] == cards[i + 1] && cards[i + 1] == cards[i + 2]) break;
                }
                if (i == 2) {
                    std::swap(cards[0], cards[3]);
                    std::swap(cards[1], cards[4]);
                }
                break;
            }
            case cgTHREE_TAKE_ONE_LINE: {
                int i;
                for (i = 0; i < cards.size() - 1; i++) {
                    if (cards[i] == cards[i + 1]) break;
                }
                int seq_length = cards.size() / 4 * 3;
                vector<int> tmp = cards;
                if (i > 0) { // need swap
                    std::copy(cards.begin() + i, cards.begin() + i + seq_length, tmp.begin());
                    std::copy(cards.begin(), cards.begin() + i, tmp.begin() + seq_length);
                }
                cards = tmp;
                break;
            }
            case cgTHREE_TAKE_TWO_LINE: {
                int i;
                for (i = 0; i < cards.size() - 2; i++) {
                    if (cards[i] == cards[i + 1] && cards[i + 1] == cards[i + 2]) break;
                }
                int seq_length = cards.size() / 5 * 3;
                vector<int> tmp = cards;
                if (i > 0) { // need swap
                    std::copy(cards.begin() + i, cards.begin() + i + seq_length, tmp.begin());
                    std::copy(cards.begin(), cards.begin() + i, tmp.begin() + seq_length);
                }
                cards = tmp;
                break;
            }
            case cgFOUR_TAKE_ONE: {
                int i;
                for (i = 0; i < 5; i++) {
                    if (cards[i] == cards[i + 1]) break;
                }
                if (i == 1) {
                    std::swap(cards[0], cards[4]);
                } else if (i == 2) {
                    std::swap(cards[0], cards[4]);
                    std::swap(cards[1], cards[5]);
                }
                break;
            }
            default:
                break;
        }
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
//        std::sort(put_list.begin(), put_list.end());
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

        SendCards(*clsGameSituation, *uctALLCardsList, this->g);
        
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
        int max_val = 0;
        int val = 0;
        clsGameSituation->nNowLandScore = LandScore(*clsGameSituation, arrHandCardData[0], val);
        clsGameSituation->nNowDiZhuID = 0;
//        for (int i = 0; i < 3; i++)
//        {
//            int val = 0;
//            int  tmpLandScore = LandScore(*clsGameSituation, arrHandCardData[i], val);
////            py::print("分为：", tmpLandScore, "sep"_a="");
//            if (tmpLandScore >= clsGameSituation->nNowLandScore && val > max_val)
//            {
//                max_val = val;
//                clsGameSituation->nNowLandScore = tmpLandScore;
//                clsGameSituation->nNowDiZhuID = i;
//            }
//        }
        
        if (clsGameSituation->nNowDiZhuID == -1)
        {
            // py::print("No one calls for 地主");
            reset();
            return prepare();
        }
        
//        py::print(clsGameSituation->nNowDiZhuID, "号是地主，分为：", clsGameSituation->nNowLandScore, "sep"_a="");
        clsGameSituation->nDiZhuID=clsGameSituation->nNowDiZhuID;
        clsGameSituation->nLandScore =clsGameSituation->nNowLandScore;
        
        
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[0]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[1]);
        arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList.push_back(clsGameSituation->DiPai[2]);
        
        arrHandCardData[clsGameSituation->nDiZhuID].Init();
        
        indexID = clsGameSituation->nDiZhuID;
        
        // py::print();
        
        
//         py::print("0号玩家牌为：");
//         arrHandCardData[0].PrintAll();
        // py::print("1号玩家牌为：");
        // arrHandCardData[1].PrintAll();
        // py::print("2号玩家牌为：");
        // arrHandCardData[2].PrintAll();  

        clsGameSituation->nCardDroit = indexID;
    }

    int prepare_manual(py::array_t<int> pycards) {
        SendCards_manual(pycards, *clsGameSituation, *uctALLCardsList, this->g);
        
        arrHandCardData[0].color_nHandCardList = (*uctALLCardsList).arrCardsList[0];
        arrHandCardData[1].color_nHandCardList = (*uctALLCardsList).arrCardsList[1];
        arrHandCardData[2].color_nHandCardList = (*uctALLCardsList).arrCardsList[2];
        
        for (int i = 0; i < 3; i++)
        {
            arrHandCardData[i].Init();
            arrHandCardData[i].nOwnIndex = i;
        }


        // call for lord
        int max_val = 0;
        for (int i = 0; i < 3; i++)
        {
            int val = 0;
            int  tmpLandScore = LandScore(*clsGameSituation, arrHandCardData[i], val);
            if (tmpLandScore >= clsGameSituation->nNowLandScore && val > max_val)
            {
                max_val = val;
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
            if (color > 52) color = 56;
            int unordered_color = color / 4 * 4;
            if (unordered_color > 52) unordered_color = 53;
            while (result[unordered_color++] > 0);
            result[unordered_color - 1]++;
        }
        return result;
    }

    std::vector<int> toOneHot60(const std::vector<int>& v) {
        std::vector<int> result(60, 0);
        for (auto color : v) {
            if (color > 52) color = 56;
            int unordered_color = color / 4 * 4;
            while (result[unordered_color++] > 0);
            result[unordered_color - 1]++;
        }
        return result;
    }

    void normalize(std::vector<int>& v, int l, int h) {
        for (int i = l; i < h; i += 4) {
            int cnt = 0;
            for (int j = i; j < i + 4; j++) {
                cnt += v[j];
            }
            for (int j = i; j < i + 4; j++) {
                if (cnt > 0) {
                    v[j] = 1;
                    cnt--;
                } else {
                    v[j] = 0;
                }
            }
        }
    }

    // one hot presentation: self_cards, remain_cards, [history 0-2], extra_cards
    // 54 * 6 dimension
    py::array_t<int> getState() {
        std::vector<int> state;
        std::vector<int> total(54, 1);

        std::vector<int> remains = total;

        // normalize history order
        vector<int> history = arrHandCardData[indexID].color_nHandCardList;
        history += clsGameSituation->color_aUnitOutCardList[indexID];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3];
        remains = remains - toOneHot(history);
//        vector<int> history[3] = {
//            toOneHot(clsGameSituation->color_aUnitOutCardList[indexID]),
//            toOneHot(clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3]),
//            toOneHot(clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3])
//        };
//        for (int i = 0; i < 3; i++) {
//            remains = remains - history[i];
//        }
//        normalize(remains, 0, 52);

        vector<int> extra_cards(std::begin(clsGameSituation->DiPai), std::end(clsGameSituation->DiPai));
        extra_cards = toOneHot(extra_cards);
        state += toOneHot(arrHandCardData[indexID].color_nHandCardList);
        state += remains;
        state += toOneHot(history);
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

    py::array_t<int> getStatePadded() {
        std::vector<int> state;
        std::vector<int> total(60, 1);
        total[53] = total[54] = total[55] = 0;
        total[57] = total[58] = total[59] = 0;

        std::vector<int> remains = total;

        // normalize history order
        vector<int> history = arrHandCardData[indexID].color_nHandCardList;
        history += clsGameSituation->color_aUnitOutCardList[indexID];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3];
        remains = remains - toOneHot60(history);
//        vector<int> history[3] = {
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[indexID]),
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3]),
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3])
//        };
//        for (int i = 0; i < 3; i++) {
//            remains = remains - history[i];
//        }
//        normalize(remains, 0, 60);

        vector<int> extra_cards(std::begin(clsGameSituation->DiPai), std::end(clsGameSituation->DiPai));
        extra_cards = toOneHot60(extra_cards);
        state += toOneHot60(arrHandCardData[indexID].color_nHandCardList);
        state += remains;
        state += toOneHot60(history);
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

    // since it's a global state, we need to keep card states in order: lord, farmer next to lord, farmer before lord
    py::array_t<int> getStateAllCards() {
        auto state = toOneHot60(arrHandCardData[clsGameSituation->nDiZhuID].color_nHandCardList);
        state += toOneHot60(arrHandCardData[(clsGameSituation->nDiZhuID + 1) % 3].color_nHandCardList);
        state += toOneHot60(arrHandCardData[(clsGameSituation->nDiZhuID + 2) % 3].color_nHandCardList);
        auto result = py::array_t<int>(state.size());
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;

        for (int i = 0; i < state.size(); ++i)
        {
            ptr[i] = state[i];
        }
        return result;
    }

    // 输出另外两个人手牌的概率
    py::array_t<float> getStateProb() {
        std::vector<float> state;
        std::vector<int> total(60, 1);
        total[53] = total[54] = total[55] = 0;
        total[57] = total[58] = total[59] = 0;

        std::vector<int> remains = total;

        // normalize history order
        vector<int> history = arrHandCardData[indexID].color_nHandCardList;
        history += clsGameSituation->color_aUnitOutCardList[indexID];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3];
        history += clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3];
        remains = remains - toOneHot60(history);
//        vector<int> history[3] = {
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[indexID]),
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[(indexID + 1) % 3]),
//            toOneHot60(clsGameSituation->color_aUnitOutCardList[(indexID + 2) % 3])
//        };
//        for (int i = 0; i < 3; i++) {
//            remains = remains - history[i];
//        }
//        normalize(remains, 0, 60);

        vector<int> extra_cards(std::begin(clsGameSituation->DiPai), std::end(clsGameSituation->DiPai));
        extra_cards = toOneHot60(extra_cards);

        vector<float> prob1(remains.begin(), remains.end());
        vector<float> prob2(remains.begin(), remains.end());
        int size1 = arrHandCardData[(indexID + 1) % 3].color_nHandCardList.size();
        int size2 = arrHandCardData[(indexID + 2) % 3].color_nHandCardList.size();
        for (int i = 0; i < remains.size(); i++) {
            prob1[i] *= float(size1) / (size1 + size2);
            prob2[i] *= float(size2) / (size1 + size2);
        }

        // divide by the other two
        // for simplicity, scale by two
        // not correct with regard to the history information, disabled now
//        for (int i = 0; i < remains.size(); i++) {
//            if (remains[i] > 0 && indexID != clsGameSituation->nDiZhuID && extra_cards[i] == 1) {
//                if (indexID + 1 == clsGameSituation->nDiZhuID) {
//                    prob1[i] = 1.f;
//                    prob2[i] = 0;
//                } else {
//                    prob1[i] = 0;
//                    prob2[i] = 1.f;
//                }
//            }
//        }

//        state += self_cards;
        state += prob1;
        state += prob2;

        auto result = py::array_t<float>(state.size());
        auto buf = result.request();
        float *ptr = (float*)buf.ptr;

        for (int i = 0; i < state.size(); ++i)
        {
            ptr[i] = state[i];
        }
        return result;
    }

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

    // 获得上家出的牌，如果自己控手，返回空
    auto getLastTwoCards() {
        vector<vector<int>> last_two_cards;
        for (int i = 2; i > 0; i--) {
            auto cards = arrHandCardData[(indexID + i) % 3].value_nPutCardList;
            last_two_cards.push_back(cards);
        }
        return last_two_cards;
//        auto last_two_cards = new vector<py::array_t<int>>();
//
//        for (int i = 2; i > 0; i--) {
//            auto cards = arrHandCardData[(indexID + i) % 3].value_nPutCardList;
//            auto result = py::array_t<int>(cards.size());
//            auto buf = result.request();
//            int *ptr = (int*)buf.ptr;
//
//            for (int j = 0; j < value_lastCards.size(); ++j)
//            {
//                ptr[j] = value_lastCards[j];
//            }
//            last_two_cards->push_back(result);
//        }
//
//        auto capsule = py::capsule(last_two_cards, [](void *v) { delete reinterpret_cast<vector<py::array_t<int>>*>(v); });
//
//        return py::array(last_two_cards->size(), last_two_cards->data(), capsule);
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
        int category_idx = static_cast<int>(category);

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
                return std::make_tuple(-1, true, category_idx);
            }
            else
            {
                // py::print("农民 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(1, true, category_idx);
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

    auto step_manual(py::array_t<int> cards = py::array_t<int>(), int card_type = -1) {
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
//            auto cg = ins_SurCardsType(cnt);
            if (card_type == -1) {
                arrHandCardData[indexID].uctPutCardType = ins_SurCardsType(cnt);
            } else {
                arrHandCardData[indexID].uctPutCardType.cgType = static_cast<CardGroupType>(card_type);
            }
//            arrHandCardData[indexID].uctPutCardType.cgType = cg.cgType;
//            if (static_cast<int>(arrHandCardData[indexID].uctPutCardType.cgType) != card_type) {
//                cout << static_cast<int>(arrHandCardData[indexID].uctPutCardType.cgType) << ", " << card_type << endl;
//            }
        }
        arrHandCardData[indexID].PutCards();
        clsGameSituation->color_aUnitOutCardList[indexID] += arrHandCardData[indexID].color_nPutCardList;

        // get group category
        int category_idx = arrHandCardData[indexID].uctPutCardType.cgType;

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
                return std::make_tuple(-1, true, category_idx);
            }
            else
            {
                // py::print("农民 ", indexID, " wins", "sep"_a="");
                return std::make_tuple(1, true, category_idx);
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
        auto category = data.uctPutCardType.cgType;
        auto intention = data.value_nPutCardList;
        reorder_cards(intention, category);
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
        reorder_cards(intention, category);
        int category_idx = static_cast<int>(category);

        // check for bomb
        bool bomb = false;
        if (intention.size() == 4) {
            if (intention[0] == intention[3]) bomb = true;
        } else if (intention.size() == 2) {
            if ((intention[0] == 16 && intention[1] == 17) || (intention[0] == 17 && intention[1] == 16)) bomb = true;
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
                return std::make_tuple(vector2numpy(intention), -1, category_idx);
            }
            else
            {
                return std::make_tuple(vector2numpy(intention), 1, category_idx);
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
        .def(py::init<long long int>(), py::arg("seed") = -1)
        .def("reset", &Env::reset)
        .def("prepare", &Env::prepare)
        .def("prepare_manual", &Env::prepare_manual)
        .def("prepare2", &Env::prepare2)
        .def("prepare2_manual", &Env::prepare2_manual)
        .def("get_state", &Env::getState)
        .def("get_state_padded", &Env::getStatePadded)
        .def("get_state_prob", &Env::getStateProb)
        .def("get_state_all_cards", &Env::getStateAllCards)
        .def("get_state2", &Env::getState2)
        .def("step", &Env::step, py::arg("lord") = false, py::arg("cards") = py::array_t<int>())
        .def("step_manual", &Env::step_manual, py::arg("cards") = py::array_t<int>(), py::arg("card_type") = -1)
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
        .def("get_last_two_cards", &Env::getLastTwoCards)
        .def("get_last_outcategory_idx", &Env::getLastCategory)
        .def("get_lord_cnt", &Env::getLordCnt);
    m.def("get_combinations_recursive", &get_combinations_recursive);
    m.def("get_combinations_nosplit", &get_combinations_nosplit);
}