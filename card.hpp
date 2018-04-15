#ifndef card_hpp
#define card_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <ctime>
using namespace std;



#define HandCardMaxLen 20



#define MinCardsValue -25



#define MaxCardsValue 106



enum CardGroupType
{
    cgERROR = -1,                                   
    cgZERO = 0,                                     
    cgSINGLE = 1,                                   
    cgDOUBLE = 2,                                   
    cgTHREE = 3,
    cgBOMB_CARD = 4,
    cgTHREE_TAKE_ONE = 5,
    cgTHREE_TAKE_TWO = 6,
    cgSINGLE_LINE = 7,
    cgDOUBLE_LINE = 8,
    cgTHREE_LINE = 9,
    cgTHREE_TAKE_ONE_LINE = 10,
    cgTHREE_TAKE_TWO_LINE = 11,
    cgKING_CARD = 12,
    cgFOUR_TAKE_ONE = 13,
    cgFOUR_TAKE_TWO = 14,

};


struct HandCardValue
{
    int SumValue;        
    int NeedRound;       
};

struct CardGroupData
{

    CardGroupType cgType=cgERROR;
    int  nValue=0;
    int  nCount=0;
    int nMaxCard=0;
    
};


class HandCardData
{
    
    
public:
    HandCardData()
    {
    }
    ~HandCardData()
    {
    }
    
public:
    vector <int> value_nHandCardList;
    
    int value_aHandCardList[18] = { 0 };
    
    vector <int> color_nHandCardList;
    int nHandCardCount = 17 ;
    int nGameRole = -1;
    int nOwnIndex = -1;
    CardGroupData uctPutCardType;
    vector <int> value_nPutCardList;
    vector <int> color_nPutCardList;
    
    HandCardValue uctHandCardValue;
public:
    
    void ClearPutCardList();
    
    void SortAsList(vector <int> &arr);
    
    bool PutOneCard(int value_nCard, int &clear_nCard);
    
    bool PutCards();
    
    void get_valueHandCardList();
    
    void Init();
    
    void PrintAll();
};

class GameSituation;

void get_PutCardList_2(GameSituation &clsGameSituation, HandCardData &clsHandCardData);
void get_PutCardList_2_limit(GameSituation &clsGameSituation, HandCardData &clsHandCardData);
void get_PutCardList_2_unlimit(HandCardData &clsHandCardData);
HandCardValue get_HandCardValue(HandCardData &clsHandCardData);
string get_CardsName(const int& card);
CardGroupData ins_SurCardsType(int arr[]);
CardGroupData ins_SurCardsType(vector<int>);
CardGroupData get_GroupData(CardGroupType cgType, int MaxCard, int Count);

#endif /* card_hpp */
