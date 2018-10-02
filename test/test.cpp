#include"../card.hpp"
#include<iostream>
#include<vector>
#include<iterator>

template <typename T> void my_cout(T any) {
    cout << any << endl;
}

float get_remain_cards_value(int cardData[], float value) {
    // cardData is an one-hot representation of cards
    bool return_flag = true;
    for(int i = 0; i < 15; i++) {
        if(cardData[i] != 0) {
            return_flag = false;
            break;
        }
    }
    if(return_flag) return value;
    vector<CardGroup> all_actions = get_all_actions(cardData);
    vector<float> value_caches;
    for(CardGroup action:all_actions) {
        float temp_group_value = get_card_group_value(action);
        value += temp_group_value;
        vector<int> cards = one_card_group2vector(action);
        if(!cards.size()) continue;
        // delete used value
        int temp_cardData[15] = {0};
        for(int j = 0; j < 15; j++) {
            int times = count(cards.begin(), cards.end(), j);
            temp_cardData[j] = cardData[j] - times;
            assert(temp_cardData[j] >= 0);
        }
        float temp_value = get_remain_cards_value(temp_cardData, value);
        value_caches.push_back(temp_value);
    }
    return *max_element(value_caches.begin(), value_caches.end());
}

CardGroupNode find_best_group(int cardData[], CardGroupType cg_type) {
    vector<CardGroup> card_groups = get_all_actions(cardData); // 0-14
    vector<float> value_caches;
    for(CardGroup card_group:card_groups) {
        if((int) card_group._category == (int) cg_type) {
            vector<int> card_group_vector = one_card_group2vector(card_group);
            float value = get_card_group_value(card_group);
            int temp_cardData[15] = {0};
            for(int j = 0; j < 15; j++) {
                int times = count(card_group_vector.begin(), card_group_vector.end(), j);
                temp_cardData[j] = cardData[j] - times;
                assert(temp_cardData[j] >= 0);
            }
            value = get_remain_cards_value(temp_cardData, value);
            value_caches.push_back(value);
        }
        else value_caches.push_back(-1000);
    }
    // after find best group
    CardGroupNode res_node;
    res_node.group_type = cg_type;
    int max_index = distance(value_caches.begin(), max_element(value_caches.begin(), value_caches.end()));
    CardGroup best_card_group = card_groups[max_index];
    vector<int> best_card_group_vector = one_card_group2vector(best_card_group);
    vector<int> remain_cards_vector;
    for(int j = 0; j < 15; j++) {
        int times = count(best_card_group_vector.begin(), best_card_group_vector.end(), j);
        int remain_times = cardData[j] - times;
        if(remain_times) {
            for (int k = 0; k < remain_times; k++) remain_cards_vector.push_back(j + 3);
        }
    }
    for(vector<int>::iterator it = best_card_group_vector.begin(); it != best_card_group_vector.end(); it ++) *it += 3;
    res_node.group_data = best_card_group_vector; 
    res_node.remain_cards = remain_cards_vector;
    return res_node;
}

template<typename T> void cout_vector(vector<T>& vec) {
    for(T v:vec) cout << v << endl;
}

int main(int argc, char const *argv[])
{
    int cardData[15] = {};
    cardData[11] = 3;
    cardData[10] = 2;
    cardData[9] = 1;
    float value = 0, r_value;
    CardGroupNode best_node = find_best_group(cardData, cgTHREE_TAKE_ONE);
    vector<int> best_group = best_node.group_data;
    vector<int> remain_cards = best_node.remain_cards;
    cout_vector(best_group);
    cout << endl;
    cout_vector(remain_cards);
    // vector<int> a = {1, 2, 3, 1, 5};
    // int dis = distance(a.begin(), max_element(a.begin(), a.end()));
    // cout << dis << endl;
    return 0;
}




float get_card_group_value(CardGroup card_group) {
    vector<int> cards = one_card_group2vector(card_group);
    Category category = card_group._category;
    float value = 0.0f;
    // empty
    if(category == Category(0)) value = 0;
    // single
    else if(category == Category(1)) {
        assert(cards.size() == 1);
        value = cards[0] * 0.1;
    } 
    // double 
    else if(category == Category(2)) {
        assert(cards.size() == 2);
        value = cards[0] * 0.2;
    } 
    // triple
    else if(category == Category(3)) {
        assert(cards.size() == 3);
        value = cards[0] * 0.3;
    } 
    // quatric
    else if(category == Category(4)) {
        assert(cards.size() == 4);
        value = cards[0] * 3;
    } 
    // three one
    else if(category == Category(5)) {
        assert(cards.size() == 4);
        int main_card, kicker;
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n == 1) kicker = card;
            else main_card = card;
        }
        value = main_card * 0.2 * 3 - kicker * 0.1;
    }
    // three two
    else if(category == Category(6)) {
        assert(cards.size() == 5);
        int main_card, kicker;
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n == 2) kicker = card;
            else main_card = card;
        }
        value = main_card * 0.2 * 3 - kicker * 0.1 * 2;
    }
    // single line
    else if(category == Category(7)) {
        assert(cards.size() >= 5);
        for(int card:cards) value += card * 0.2;
    }
    // double line
    else if(category == Category(8)) {
        assert(cards.size() >= 6);
        for(int card:cards) value += card * 0.2;
    }
    // triple line
    else if(category == Category(9)) {
        assert(cards.size() == 6);
        for(int card:cards) value += card * 0.2;
    }
    // three one line
    else if(category == Category(10)) {
        assert(cards.size() == 8);
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n <= 2) value += -0.1 * card;
            else value += 0.1 * card;
        }
    }
    // three two line 
    else if(category == Category(11)) {
        assert(cards.size() == 10);
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n <= 2) value += -0.1 * card;
            else value += 0.1 * card;
        }
    }
    // big band
    else if(category == Category(12)) value = 5;
    // four take one
    else if(category == Category(13)) {
        assert(cards.size() == 5);
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n == 1) value += -0.1 * card;
            else value += 0.1 * card;
        }
    }
    // take take two
    else if(category == Category(14)) {
        assert(cards.size() == 6);
        for(int card:cards) {
            int count_n = count(cards.begin(), cards.end(), card);
            if(count_n == 1) value += -0.1 * card;
            else value += 0.1 * card;
        }
    }
    return value;
} 

vector<int> one_card_group2vector(CardGroup card_group) {
    vector<int> vct;
    vector<Card> cg_cards = card_group._cards;
    for(Card cd:cg_cards) {
            vct.push_back((int) cd);
        }
    return vct;
}

void get_one_hot_respresentation(int one_hot[], vector<int> hand_card_data, bool zero_start) {
    for(int idx = 0; idx < 15; idx ++) {
        int new_idx = idx;
        if(!zero_start) new_idx = idx + 3;
        for(int card:hand_card_data) {
            if(new_idx == card) one_hot[idx] += 1;
        }
    }
    return;
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
