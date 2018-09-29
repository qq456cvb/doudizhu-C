#include "card.h"
#include <iostream>
#include <iterator>
#include <algorithm>
#include <assert.h>


// Nuke can not be used as kicker, different from Python side, 
// TODO: modify the Python side
void get_kickers(const vector<Card> main_cards, bool single, int len, vector<vector<Card>> &kickers) {
	if (len == 0)
	{
		return;
	}
	vector<vector<Card>> result;
	for (auto &kicker : kickers) {
		for (size_t i = (kicker.empty() ? 0 : static_cast<int>(kicker.back()) + 1); i < (single ? 15 : 13); i++)
		{
			vector<Card> tmp = kicker;
			if (find(main_cards.begin(), main_cards.end(), Card(i)) == main_cards.end())
			{
				tmp.push_back(Card(i));
				if (!single)
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
	get_kickers(main_cards, single, len - 1, kickers);
}

//************************************
// Method:    get_all_actions
// FullName:  get_all_actions
// Access:    public 
// Returns:   std::vector<CardGroup>
// Qualifier: the action orders may not be the same as Python's
//************************************
vector<CardGroup> get_all_actions() {
	vector<CardGroup> actions;
	actions.push_back(CardGroup({}, Category::EMPTY, 0));
	for (int i = 0; i < 15; i++)
	{
		actions.push_back(CardGroup({ Card(i) }, Category::SINGLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
		actions.push_back(CardGroup({ Card(i), Card(i) }, Category::DOUBLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
		actions.push_back(CardGroup({ Card(i), Card(i), Card(i) }, Category::TRIPLE, i));
	}
	for (int i = 0; i < 13; i++)
	{
		vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
		actions.push_back(CardGroup(main_cards, Category::QUADRIC, i));
		vector<vector<Card>> kickers = { {} };
		get_kickers(main_cards, true, 2, kickers);
		for (const auto &kicker : kickers)
		{
			auto cards_kickers = main_cards;
			cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
			actions.push_back(CardGroup(cards_kickers, Category::FOUR_TAKE_ONE, i));
		}
		kickers.clear();
		kickers.push_back({});
		get_kickers(main_cards, false, 2, kickers);
		for (const auto &kicker : kickers)
		{
			auto cards_kickers = main_cards;
			cards_kickers.insert(cards_kickers.end(), kicker.begin(), kicker.end());
			actions.push_back(CardGroup(cards_kickers, Category::FOUR_TAKE_TWO, i));
		}
	}
	for (int i = 0; i < 13; i++)
	{
		for (int j = 0; j < 15; j++)
		{
			if (i != j) {
				actions.push_back(CardGroup({ Card(i), Card(i), Card(i), Card(j) }, Category::THREE_ONE, i));
				if (j < 13)
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
			get_kickers(cards, true, len, kickers);
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
				get_kickers(cards, false, len, kickers);
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
	actions.push_back(CardGroup({ Card(13), Card(14) }, Category::BIGBANG, 100));
	return actions;
}

auto all_actions = get_all_actions();

ostream& operator<<(ostream& os, const Card& c) {
	if (c == Card::THREE)
	{
		os << "3";
	} 
	else if (c == Card::FOUR)
	{
		os << "4";
	}
	else if (c == Card::FIVE)
	{
		os << "5";
	}
	else if (c == Card::SIX)
	{
		os << "6";
	}
	else if (c == Card::SEVEN)
	{
		os << "7";
	}
	else if (c == Card::EIGHT)
	{
		os << "8";
	}
	else if (c == Card::NINE)
	{
		os << "9";
	}
	else if (c == Card::TEN)
	{
		os << "X";
	}
	else if (c == Card::JACK)
	{
		os << "J";
	}
	else if (c == Card::QUEEN)
	{
		os << "Q";
	}
	else if (c == Card::KING)
	{
		os << "K";
	}
	else if (c == Card::ACE)
	{
		os << "A";
	}
	else if (c == Card::TWO)
	{
		os << "2";
	}
	else if (c == Card::BLACK_JOKER)
	{
		os << "*";
	}
	else if (c == Card::RED_JOKER)
	{
		os << "$";
	}
	return os;
}

ostream& operator<<(ostream& os, const CardGroup& cg) {
	for (auto c : cg._cards)
	{
		os << c << ", ";
	}
	return os;
}
