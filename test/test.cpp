// #include"../card.hpp"
// #include<iostream>
// #include<vector>
// #include<iterator>
// #include<fstream>
# include<iostream>
using namespace std;
// template <typename T> void my_cout(T any) {
//     cout << any << endl;
// }
// template<typename T> void cout_vector(vector<T> any) {
//     for(T a:any) my_cout(a);
// }

// float get_remain_cards_value(int cardData[], float value) {
//     // cardData is an one-hot representation of cards
//     bool return_flag = true;
//     for(int i = 0; i < 15; i++) {
//         if(cardData[i] != 0) {
//             return_flag = false;
//             break;
//         }
//     }
//     if(return_flag) return value;
//     int max_value = 0;
//     for(int max_idx = 14; max_idx > -1; max_idx --) {
//         if(cardData[max_idx] > 0) {
//             max_value = max_idx;
//             break;
//         }
//     }

//     vector<CardGroup> all_actions = get_all_actions_unlimit(cardData);
//     vector<float> value_caches;
//     for(CardGroup action:all_actions) {
//         if(!action._cards.size()) continue;
//         vector<int> cards = one_card_group2vector(action);
//         assert(cards.size() > 0);
//         if(find(cards.begin(), cards.end(), max_value) == cards.end()) continue;
//         float temp_group_value = get_card_group_value(action);

//         value += temp_group_value;
//         // delete used value
//         int temp_cardData[15] = {0};
//         for(int j = 0; j < 15; j++) {
//             int times = count(cards.begin(), cards.end(), j);
//             temp_cardData[j] = cardData[j] - times;

//             assert(temp_cardData[j] >= 0);
//         }
//         float temp_value = get_remain_cards_value(temp_cardData, value);
//         value_caches.push_back(temp_value);
//     }
//     return *max_element(value_caches.begin(), value_caches.end());
// }
// void get_kickers(const vector<Card> main_cards, bool single, int len, vector<vector<Card>> &kickers, int cardData[]) {
// 	if (len == 0)
// 	{
// 		return;
// 	}
// 	vector<vector<Card>> result;
// 	for (auto &kicker : kickers) {
// 		for (size_t i = (kicker.empty() ? 0 : static_cast<int>(kicker.back()) + 1); i < (single ? 15 : 13); i++)
// 		{
// 			vector<Card> tmp = kicker;
// 			if (find(main_cards.begin(), main_cards.end(), Card(i)) == main_cards.end() && cardData[i] >= 1)
// 			{
//                 tmp.push_back(Card(i + 3));
//                 if (!single && cardData[i] >= 2)
//                 {
//                     tmp.push_back(Card(i + 3));
//                 }
//                 if (!(tmp.size() == 2 && static_cast<int>(tmp[0]) + static_cast<int>(tmp[1]) == 13 + 14))
//                 {
//                     result.push_back(tmp);
//                 }
// 			}
// 		}
// 	}
// 	kickers = result;
// 	get_kickers(main_cards, single, len - 1, kickers, cardData);
// }
// vector<int> one_card_group2vector(CardGroup card_group) {
//     vector<int> vct = {};
//     vector<Card> cg_cards = card_group._cards;
//     if(!cg_cards.size()) return vct;
//     for(Card cd:cg_cards) {
//             vct.push_back((int) cd);
//         }
//     return vct;
// }
// float get_card_group_value(CardGroup card_group) {
//     vector<int> cards = one_card_group2vector(card_group);
//     Category category = card_group._category;
//     float value = 0.0f;
//     // empty
//     if(category == Category(0)) value = 0;
//     // single
//     else if(category == Category(1)) {
//         assert(cards.size() == 1);
//         value = cards[0] * 0.3;
//     }
//     // double
//     else if(category == Category(2)) {
//         assert(cards.size() == 2);
//         value = cards[0] * 0.4;
//     }
//     // triple
//     else if(category == Category(3)) {
//         assert(cards.size() == 3);
//         value = cards[0] * 0.8;
//     }
//     // quatric
//     else if(category == Category(4)) {
//         assert(cards.size() == 4);
//         value = cards[0] * 3;
//     }
//     // three one
//     else if(category == Category(5)) {
//         assert(cards.size() == 4);
//         int main_card, kicker;
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n == 1) kicker = card;
//             else main_card = card;
//         }
//         value = main_card * 0.2 * 3 - kicker * 0.1;
//     }
//     // three two
//     else if(category == Category(6)) {
//         assert(cards.size() == 5);
//         int main_card, kicker;
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n == 2) kicker = card;
//             else main_card = card;
//         }
//         value = main_card * 0.2 * 3 - kicker * 0.1 * 2;
//     }
//     // single line
//     else if(category == Category(7)) {
//         assert(cards.size() >= 5);
//         for(int card:cards) value += card * 0.2;
//     }
//     // double line
//     else if(category == Category(8)) {
//         assert(cards.size() >= 6);
//         for(int card:cards) value += card * 0.2;
//     }
//     // triple line
//     else if(category == Category(9)) {
//         assert(cards.size() % 3 == 0);
//         for(int card:cards) value += card * 0.2;
//     }
//     // three one line
//     else if(category == Category(10)) {
//         assert(cards.size() >= 3);
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n <= 2) value += -0.1 * card;
//             else value += 0.1 * card;
//         }
//     }
//     // three two line
//     else if(category == Category(11)) {
//         assert(category == Category::THREE_TWO_LINE);
//         assert(cards.size() >= 3);
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n <= 2) value += -0.1 * card;
//             else value += 0.1 * card;
//         }
//     }
//     // big band
//     else if(category == Category(12)) value = 5;
//     // four take one
//     else if(category == Category(13)) {
//         assert(cards.size() == 6);
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n == 1) value += -0.1 * card;
//             else value += 0.1 * card;
//         }
//     }
//     // take take two
//     else if(category == Category(14)) {
//         assert(cards.size() == 8);
//         for(int card:cards) {
//             int count_n = count(cards.begin(), cards.end(), card);
//             if(count_n == 1) value += -0.1 * card;
//             else value += 0.1 * card;
//         }
//     }
//     return value;
// }
// vector<CardGroup> get_all_actions_unlimit(int cardData[]) {
// 	vector<CardGroup> actions;
//     // actions.push_back(CardGroup({}, Category::EMPTY, 0));
// 	for (int i = 0; i < 15; i++)
// 	{
//         if(cardData[i] >= 1) actions.push_back(CardGroup({ Card(i) }, Category::SINGLE, i));
// 	}
// 	for (int i = 0; i < 13; i++)
// 	{
// 		if(cardData[i] >= 2) actions.push_back(CardGroup({ Card(i), Card(i) }, Category::DOUBLE, i));
// 	}
// 	for (int i = 0; i < 13; i++)
// 	{
// 		if(cardData[i] >= 3) actions.push_back(CardGroup({ Card(i), Card(i), Card(i) }, Category::TRIPLE, i));
// 	}
// 	for (int i = 0; i < 13; i++)
// 	{
//         if(cardData[i] == 4) {
//             vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
//             actions.push_back(CardGroup(main_cards, Category::QUADRIC, i));
//             for(int idx = 0; idx < 15; idx ++)
//             {
//                 vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
//                 if(4 > cardData[idx] - count(main_cards.begin(), main_cards.end(), Card(idx)) && cardData[idx] - count(main_cards.begin(), main_cards.end(), Card(idx)) >= 1 && idx != i)

//                 {
//                     main_cards.push_back(Card(idx));
//                     cardData[idx] -= 1;
//                     for(int idx1 = 0; idx1 < 15; idx1 ++)
//                     {
//                         if(4 > cardData[idx1] - count(main_cards.begin(), main_cards.end(), Card(idx1)) && cardData[idx1] - count(main_cards.begin(), main_cards.end(), Card(idx1)) >= 1 && idx1 != i)
//                         {
//                             main_cards.push_back(Card(idx1));
//                             assert(main_cards.size() == 6);
//                             actions.push_back(CardGroup(main_cards, Category::FOUR_TAKE_ONE, i));
//                             main_cards.pop_back();
//                         }
//                     }
//                     cardData[idx] += 1;
//                 }
//             }
//             for(int idx = 0; idx < 15; idx ++)
//             {
//                 vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
//                 if(4 > cardData[idx] - count(main_cards.begin(), main_cards.end(), Card(idx)) && cardData[idx] - count(main_cards.begin(), main_cards.end(), Card(idx)) >= 2 && idx != i)

//                 {
//                     main_cards.push_back(Card(idx));
//                     main_cards.push_back(Card(idx));
//                     cardData[idx] -= 2;
//                     for(int idx1 = 0; idx1 < 15; idx1 ++)
//                     {
//                         if(4 > cardData[idx1] - count(main_cards.begin(), main_cards.end(), Card(idx1)) && cardData[idx1] - count(main_cards.begin(), main_cards.end(), Card(idx1)) >= 2 && idx1 != i)
//                         {
//                             main_cards.push_back(Card(idx1));
//                             main_cards.push_back(Card(idx1));
//                             assert(main_cards.size() == 8);
//                             actions.push_back(CardGroup(main_cards, Category::FOUR_TAKE_TWO, i));
//                             main_cards.pop_back();
//                             main_cards.pop_back();

//                         }
//                     }
//                     cardData[idx] += 2;
//                 }
//             }

//         }
// 	}
// 	for (int i = 0; i < 13; i++)
// 	{
// 		for (int j = 0; j < 15; j++)
// 		{
// 			if (i != j && cardData[i] >= 3 && cardData[j] >= 1) {
// 				actions.push_back(CardGroup({ Card(i), Card(i), Card(i), Card(j) }, Category::THREE_ONE, i));
// 				if (j < 13 && cardData[j] >= 2)
// 				{
// 					actions.push_back(CardGroup({ Card(i), Card(i), Card(i), Card(j), Card(j) }, Category::THREE_TWO, i));
// 				}
// 			}
// 		}
// 	}
// 	for (size_t i = 0; i < 12; i++)
// 	{
// 		for (size_t j = i + 5; j < 13; j++)
// 		{
//             bool stop_flag = false;
//             for(size_t l = i; l < j; l++) {
//                 if(cardData[l] == 0) stop_flag = true;
//             }
//             if(stop_flag) break;
// 			vector<Card> cards;
// 			for (size_t k = i; k < j; k++)
// 			{
// 				cards.push_back(Card(k));
// 			}
// 			assert(cards.size() >= 5);
// 			actions.push_back(CardGroup(cards, Category::SINGLE_LINE, i, cards.size()));
// 		}
// 		for (size_t j = i + 3; j < 13; j++)
// 		{
//             bool stop_flag = false;
//             for(size_t l = i; l < j; l++) {
//                 if(cardData[l] <= 1) stop_flag = true;
//             }
//             if(stop_flag) break;
// 			vector<Card> cards;
// 			for (size_t k = i; k < j; k++)
// 			{
// 				cards.push_back(Card(k));
// 				cards.push_back(Card(k));
// 			}
// 			assert(cards.size() >= 6);
// 			if (cards.size() <= 20)
// 			{
// 				actions.push_back(CardGroup(cards, Category::DOUBLE_LINE, i, cards.size() / 2));
// 			}
// 		}
// 		for (size_t j = i + 2; j < 13; j++)
// 		{
//             bool stop_flag = false;
//             for(size_t l = i; l < j; l++) {
//                 if(cardData[l] <= 2) stop_flag = true;
//             }
//             if(stop_flag) break;
// 			vector<Card> cards;
// 			for (size_t k = i; k < j; k++)
// 			{
// 				cards.push_back(Card(k));
// 				cards.push_back(Card(k));
// 				cards.push_back(Card(k));
// 			}
// 			assert(cards.size() >= 6);
// 			if (cards.size() <= 20)
// 			{
// 				actions.push_back(CardGroup(cards, Category::TRIPLE_LINE, i, cards.size() / 3));
// 			}
// 			size_t len = cards.size() / 3;
// 			if(len == 2) {
// 			    for(int idx = 0; idx < 15; idx ++) // three one line
//                 {
//                     if(3 > cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) && cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) >= 1 && cardData[idx] < 3)
//                     {
//                         cards.push_back(Card(idx));
//                         cardData[idx] -= 1;
//                         for(int idx1 = 0; idx1 < 15; idx1 ++)
//                         {
//                             if(3 > (cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1))) && (cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1))) >= 1 && cardData[idx1] < 3)
//                             {
//                                 cards.push_back(Card(idx1));
//                                 assert(cards.size() == 8);
//                                 actions.push_back(CardGroup(cards, Category::THREE_ONE_LINE, i));
//                                 cards.pop_back();
//                             }
//                         }
//                         cards.pop_back();
//                         cardData[idx] += 1;
//                     }
//                 }
//                 for(int idx = 0; idx < 15; idx ++) // three two line
//                 {
//                     if(3 > cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) && cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) >= 2 && cardData[idx] < 3)
//                     {
//                         cards.push_back(Card(idx));
//                         cards.push_back(Card(idx));
//                         cardData[idx] -= 2;
//                         for(int idx1 = 0; idx1 < 15; idx1 ++)
//                         {
//                             if(3 > cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) && cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) >= 2 && cardData[idx1] < 3)
//                             {
//                                 cards.push_back(Card(idx1));
//                                 cards.push_back(Card(idx1));
//                                 assert(cards.size() == 10);
//                                 actions.push_back(CardGroup(cards, Category::THREE_TWO_LINE, i));
//                                 cards.pop_back();
//                                 cards.pop_back();
//                             }
//                         }
//                         cards.pop_back();
//                         cards.pop_back();
//                         cardData[idx] += 2;
//                     }
//                 }
// 			}
// 			else if(len == 3) {
//                 for(int idx = 0; idx < 15; idx ++) {// three one line
//                     if(3 > cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) && cardData[idx] - count(cards.begin(), cards.end(), Card(idx))  >= 1 && cardData[idx] < 3)
//                     {
//                         cards.push_back(Card(idx));
//                         cardData[idx] -= 1;
//                         for(int idx1 = 0; idx1 < 15; idx1 ++)
//                         {
//                             if(3 > cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) && cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) >= 1 && cardData[idx1] < 3)
//                             {
//                                 cards.push_back(Card(idx1));
//                                 cardData[idx1] -= 1;
//                                 {
//                                     for(int idx2 = 0; idx2 < 15; idx2 ++)
//                                     {   // three one line
//                                         if(3 > cardData[idx2] - count(cards.begin(), cards.end(), Card(idx2)) && cardData[idx2] - count(cards.begin(), cards.end(), Card(idx2)) >= 1 && cardData[idx2] < 3)
//                                         {
//                                             cards.push_back(Card(idx2));
//                                             assert(cards.size() == 12);
//                                             actions.push_back(CardGroup(cards, Category::THREE_ONE_LINE, i));
//                                             cards.pop_back();
//                                         }
//                                      }
//                                 cards.pop_back();
//                                 cardData[idx1] += 1;
//                                 }
//                             }
//                         }
//                         cards.pop_back();
//                         cardData[idx] += 1;
//                     }
//                 }
//                 for(int idx = 0; idx < 15; idx ++) // three two line
//                 {
//                     if(3 > cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) && cardData[idx] - count(cards.begin(), cards.end(), Card(idx)) >= 2 && cardData[idx] < 3)
//                     {
//                         cards.push_back(Card(idx));
//                         cards.push_back(Card(idx));
//                         cardData[idx] -= 2;
//                         for(int idx1 = 0; idx1 < 15; idx1 ++)
//                         {
//                             if(3 > cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) && cardData[idx1] - count(cards.begin(), cards.end(), Card(idx1)) >= 2 && cardData[idx1] < 3)
//                             {
//                                 cards.push_back(Card(idx1));
//                                 cards.push_back(Card(idx1));
//                                 cardData[idx1] -= 2;
//                                 for(int idx2 = 0; idx2 < 15; idx2 ++)
//                                 {
//                                     if(3 > cardData[idx2] - count(cards.begin(), cards.end(), Card(idx2)) && cardData[idx2] - count(cards.begin(), cards.end(), Card(idx2)) >= 2 && cardData[idx2] < 3)
//                                     {
//                                         cards.push_back(Card(idx2));
//                                         cards.push_back(Card(idx2));
//                                         assert(cards.size() == 15);
//                                         actions.push_back(CardGroup(cards, Category::THREE_TWO_LINE, i));
//                                         cards.pop_back();
//                                         cards.pop_back();
//                                     }
//                                 }
//                                 cards.pop_back();
//                                 cards.pop_back();
//                                 cardData[idx1] += 2;
//                             }
//                         }
//                         cards.pop_back();
//                         cards.pop_back();
//                         cardData[idx] += 2;
//                     }
//                 }

// 			}
// 		}
// 	}
// 	if(cardData[13] != 0 && cardData[14] != 0) actions.push_back(CardGroup({ Card(13), Card(14) }, Category::BIGBANG, 100));
// 	if(!actions.size()) actions.push_back(CardGroup({}, Category::EMPTY, 0));
// 	return actions;
// }
// void get_one_hot_representation(int one_hot[], vector<int> hand_card_data, bool zero_start) {
//     for(int idx = 0; idx < 15; idx ++) {
//         int new_idx = idx;
//         if(!zero_start) new_idx = idx + 3;
//         for(int card:hand_card_data) {
//             if(new_idx == card) one_hot[idx] += 1;
//         }
//     }
//     return;
// }

int main(int argc, char const *argv[])
{
    //   float value = 0;
    //   vector<float> value_caches = {};
    //   vector<int> cardData_vector = {7, 7, 7, 7, 8, 8, 8, 8, 3, 3, 3, 4};
    //   int cardData[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    //   get_one_hot_representation(cardData, cardData_vector, false);
    //   for(int _ = 0; _ < 15; _ ++) my_cout(cardData[_]);
    //   cout << endl;
    //   vector<CardGroup> all_actions = get_all_actions_unlimit(cardData);
    //   for(int _ = 0; _ < 15; _ ++) my_cout(cardData[_]);
    //   ofstream myfile;
    //   myfile.open("exe.txt");
    //   for(CardGroup action:all_actions) {
    //         vector<int> cards = one_card_group2vector(action);
    //         cout_vector(cards);
    //         myfile << 1 << endl;
    //         cout << "length is " << cards.size() << endl;
    //         int type = (int) action._category;
    //         cout << "type is " << type << endl;
    // }
//        // if(!action._cards.size()) continue;
//        vector<int> cards = one_card_group2vector(action);
      // for(int c = 0; c < 15; c ++) my_cout(cardData[c]);
//      vector<CardGroup> all_actions = get_all_actions_unlimit(cardData);
//      int max_value = 0;
//       for(int max_idx = 14; max_idx > -1; max_idx --) {
//           if(cardData[max_idx] > 0) {
//               max_value = max_idx;
//               break;
//           }
//       }
//      for(CardGroup action:all_actions) {
//        // if(!action._cards.size()) continue;
//        vector<int> cards = one_card_group2vector(action);
//        // assert(cards.size() > 0);
//         if(find(cards.begin(), cards.end(), max_value) == cards.end()) continue;
////         cout_vector(cards);
////        cout << endl;
//         float temp_group_value = get_card_group_value(action);
//
//        value += temp_group_value;
//        // delete used value
//        int temp_cardData[15] = {0};
//        for(int j = 0; j < 15; j++) {
//            int times = count(cards.begin(), cards.end(), j);
//            temp_cardData[j] = cardData[j] - times;
//            cout_vector(cards);
//            cout << endl;
//            assert(temp_cardData[j] >= 0);
//        }
//        float temp_value = get_remain_cards_value(temp_cardData, value);
//        value_caches.push_back(temp_value);
//
//      }
//      cout_vector(value_caches);
    int a[3] = {1, 2, 3};
    int* b = a;
    cout << b[2] << endl;
    return 0;
}



