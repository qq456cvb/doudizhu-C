#ifndef card_hpp
#define card_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <ctime>
#include <assert.h>
using namespace std;

#define HandCardMaxLen 20
#define MinCardsValue -25



#define MaxCardsValue 106

/* my modification */
enum class Category {
	EMPTY = 0,
	SINGLE = 1,
	DOUBLE = 2,
	TRIPLE = 3,
	QUADRIC = 4,
	THREE_ONE = 5,
	THREE_TWO = 6,
	SINGLE_LINE = 7,
	DOUBLE_LINE = 8,
	TRIPLE_LINE = 9,
	THREE_ONE_LINE = 10,
	THREE_TWO_LINE = 11,
	BIGBANG = 12,
	FOUR_TAKE_ONE = 13,
	FOUR_TAKE_TWO = 14
};

enum class Card {
	THREE = 0,
	FOUR = 1,
	FIVE = 2,
	SIX = 3,
	SEVEN = 4,
	EIGHT = 5,
	NINE = 6,
	TEN = 7,
	JACK = 8,
	QUEEN = 9,
	KING = 10,
	ACE = 11,
	TWO = 12,
	BLACK_JOKER = 13,
	RED_JOKER = 14
};

ostream& operator<<(ostream& os, const Card& c);

class CardGroup {
public:
	CardGroup() {};
	CardGroup(const vector<Card> &cards, Category category, int rank, int len = 1)
		: _cards(cards), _category(category), _rank(rank), _len(len)
	{
	};

	vector<Card> _cards;
	Category _category;
	int _rank, _len;

	bool operator==(const CardGroup &other) const {
		return _category == other._category && _rank == other._rank;
	}
	bool operator>(const CardGroup &other) const {
		if (this->_category == Category::EMPTY) return other._category != Category::EMPTY;
		if (other._category == Category::EMPTY) return true;
		if (this->_category == Category::BIGBANG)
		{
			return true;
		}
		if (other._category == Category::BIGBANG)
		{
			return false;
		}
		if (this->_category != other._category)
		{
			if (this->_category == Category::QUADRIC)
			{
				return true;
			}
			if (other._category == Category::QUADRIC)
			{
				return false;
			}
			return false;
		}
		else {
			return this->_rank > other._rank && this->_len == other._len;
		}
	}
	friend ostream& operator<<(ostream& os, const CardGroup& cg);
};
/* my modification */


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

struct CardGroupNode {
    CardGroupType group_type;
    vector<int> group_data;
	vector<int> remain_cards;
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

struct CardValue {
    float group_value;
    float remain_card_value;
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

void get_kickers(const vector<Card> main_cards, bool single, int len, vector<vector<Card>> &kickers, int cardData[]);
vector<vector<int>> cardGroupNode2matrix(vector<CardGroupNode> &card_group_nodes);
vector<vector<int>> CardGroup2matrix(vector<CardGroup> card_group);
vector<int> one_card_group2vector(CardGroup card_group);
vector<CardGroup> get_all_actions(int cardData[]);
void get_one_hot_respresentation(int one_hot[], vector<int> hand_card_data, bool zero_start);
float get_card_group_value(CardGroup card_group);
CardGroupNode find_best_group(int cardData[], CardGroupType cg_type);
float get_remain_cards_value(int cardData[], float value);
#endif /* card_hpp */
