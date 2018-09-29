#include<iostream>
#include<vector>
#include"../card.hpp"
#include <cassert>

using namespace std;

int main(int argc, char const *argv[])
{
    // vector<vector<int>> a;
    // for(int  i = 0; i < 5; i ++) {
    //     a.push_back({2});
    // }
    // for(vector<int> x:a) {
    //     cout << x[0] << endl;
    // }
    int cardData[15] = {};
    cardData[11] = 3;
    cardData[10] = 2;
    cardData[9] = 1;
    int count = 0;
    auto all_actions = get_all_actions(cardData);
    vector<vector<int>> a = CardGroup2matrix(all_actions);
    for(vector<int> b:a) {
        for(int c:b) {
            cout << c << endl;
        }
        cout << endl;
    }
    // for(int i = 0; i < 15; i++) {
    //     cout << cardData[i] << endl;
    // }
    return 0;
}
// here cardData is a one hot representation of current hand data
vector<CardGroup> get_all_actions(int cardData[]) {
	vector<CardGroup> actions;
	actions.push_back(CardGroup({}, Category::EMPTY, 0));
	for (int i = 0; i < 15; i++)
	{
        if(cardData[i] >= 1) actions.push_back(CardGroup({ Card(i) }, Category::SINGLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
		if(cardData[i] >= 2) actions.push_back(CardGroup({ Card(i), Card(i) }, Category::DOUBLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
		if(cardData[i] >= 3) actions.push_back(CardGroup({ Card(i), Card(i), Card(i) }, Category::TRIPLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
        if(cardData[i] == 4) {
            vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
            actions.push_back(CardGroup(main_cards, Category::QUADRIC, i));
            vector<vector<Card>> kickers = { {} };
            get_kickers(main_cards, true, 2, kickers, cardData);
            for (const auto &kicker : kickers)
            {
                auto cards_kickers = main_cards;
                cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
                actions.push_back(CardGroup(cards_kickers, Category::FOUR_TAKE_ONE, i));
            }
            kickers.clear();
            kickers.push_back({});
            get_kickers(main_cards, false, 2, kickers, cardData);
            for (const auto &kicker : kickers)
            {
                auto cards_kickers = main_cards;
                cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
                actions.push_back(CardGroup(cards_kickers, Category::FOUR_TAKE_TWO, i));
            }
        }
	}
	for (int i = 0; i < 13; i++)
	{
		for (int j = 0; j < 15; j++)
		{
			if (i != j && cardData[i] >= 3 && cardData[j] >= 1) {
				actions.push_back(CardGroup({ Card(i), Card(i), Card(i), Card(j) }, Category::THREE_ONE, i));
				if (j < 13 && cardData[j] >= 2)
				{
					actions.push_back(CardGroup({ Card(i), Card(i), Card(i), Card(j), Card(j) }, Category::THREE_TWO, i));
				}
			}
		}
	}
	for (size_t i = 0; i < 12; i++)
	{
		for (size_t j = i + 5; j < 13; j++)
		{
            bool stop_flag = false;
            for(size_t l = i; l < j; l++) {
                if(cardData[l] == 0) stop_flag = true;
            }
            if(stop_flag) break;
			vector<Card> cards;
			for (size_t k = i; k < j; k++)
			{
				cards.push_back(Card(k));
			}
			assert(cards.size() >= 5);
			actions.push_back(CardGroup(cards, Category::SINGLE_LINE, i, cards.size()));
		}
		for (size_t j = i + 3; j < 13; j++)
		{
            bool stop_flag = false;
            for(size_t l = i; l < j; l++) {
                if(cardData[l] <= 1) stop_flag = true;
            }
            if(stop_flag) break;
			vector<Card> cards;
			for (size_t k = i; k < j; k++)
			{
				cards.push_back(Card(k));
				cards.push_back(Card(k));
			}
			assert(cards.size() >= 6);
			if (cards.size() <= 20)
			{
				actions.push_back(CardGroup(cards, Category::DOUBLE_LINE, i, cards.size() / 2));
			}
		}
		for (size_t j = i + 2; j < 13; j++)
		{
            bool stop_flag = false;
            for(size_t l = i; l < j; l++) {
                if(cardData[l] <= 2) stop_flag = true;
            }
            if(stop_flag) break;
			vector<Card> cards;
			for (size_t k = i; k < j; k++)
			{
				cards.push_back(Card(k));
				cards.push_back(Card(k));
				cards.push_back(Card(k));
			}
			assert(cards.size() >= 6);
			if (cards.size() <= 20)
			{
				actions.push_back(CardGroup(cards, Category::TRIPLE_LINE, i, cards.size() / 3));
			}
			size_t len = cards.size() / 3;
			vector<vector<Card>> kickers = { {} };
			get_kickers(cards, true, len, kickers, cardData);
			for (const auto &kicker : kickers)
			{
				/*std::copy(kicker.begin(), kicker.end(), std::ostream_iterator<Card>(cout));
				cout << endl;*/
				auto cards_kickers = cards;
				cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
				if (cards_kickers.size() <= 20)
				{
					actions.push_back(CardGroup(cards_kickers, Category::THREE_ONE_LINE, i, len));
				}
			}
			if (len < 5)
			{
				kickers.clear();
				kickers.push_back({});
				get_kickers(cards, false, len, kickers, cardData);
				for (const auto &kicker : kickers)
				{
					auto cards_kickers = cards;
					cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
					if (cards_kickers.size() <= 20)
					{
						actions.push_back(CardGroup(cards_kickers, Category::THREE_TWO_LINE, i, len));
					}
				}
			}
		}
	}
	if(cardData[13] != 0 && cardData[14] != 0) actions.push_back(CardGroup({ Card(13), Card(14) }, Category::BIGBANG, 100));
	return actions;
}

void get_kickers(const vector<Card> main_cards, bool single, int len, vector<vector<Card>> &kickers, int cardData[]) {
	if (len == 0)
	{
		return;
	}
	vector<vector<Card>> result;
	for (auto &kicker : kickers) {
		for (size_t i = (kicker.empty() ? 0 : static_cast<int>(kicker.back()) + 1); i < (single ? 15 : 13); i++)
		{
			vector<Card> tmp = kicker;
			if (find(main_cards.begin(), main_cards.end(), Card(i)) == main_cards.end() && cardData[i] >= 1)
			{
                tmp.push_back(Card(i));
                if (!single && cardData[i] >= 2)
                {
                    tmp.push_back(Card(i));
                }
                if (!(tmp.size() == 2 && static_cast<int>(tmp[0]) + static_cast<int>(tmp[1]) == 13 + 14))
                {
                    result.push_back(tmp);
                }           	
			}
		}
	}
	kickers = result;
	get_kickers(main_cards, single, len - 1, kickers, cardData);
}

vector<vector<int>> CardGroup2matrix(vector<CardGroup> card_group) {
    vector<vector<int>> card_group_matrix;
    vector<int> one_row; 
    for(CardGroup cg:card_group) {
        vector<Card> cg_cards = cg._cards;
        for(Card cd:cg_cards) {
            one_row.push_back((int) cd);
        }
        card_group_matrix.push_back(one_row);
        one_row.clear();
    }
    return card_group_matrix;
}
