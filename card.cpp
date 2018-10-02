#include "card.hpp"
#include "game.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

string get_CardsName(const int& card)
{
  string result = "";
  int color = card % 4;
  switch (color) {
  case 0:
    result += "草花";
    break;
  case 1:
    result += "方片";
    break;
  case 2:
    result += "黑桃";
    break;
  case 3:
    result += "红桃";
    break;

  default:
    break;
  }
  int main = card / 4;
  if (main <= 7) {
    result += to_string(main + 3);
  }
  else {
    switch (main) {
    case 8:
      result += "J";
      break;
    case 9:
      result += "Q";
      break;
    case 10:
      result += "K";
      break;
    case 11:
      result += "A";
      break;
    case 12:
      result += "2";
      break;
    case 13:
      result = "小王";
      break;
    case 14:
      result = "大王";
      break;

    default:
      break;
    }
  }
  return result;
}

/*  */

void Put_All_SurCards(GameSituation &clsGameSituation, HandCardData &clsHandCardData, CardGroupData SurCardGroupData)
{

    /**/
    for (int i = 0; i < 18; i++)
        for (int j = 0; j< clsHandCardData.value_aHandCardList[i]; j++)
            clsHandCardData.value_nPutCardList.push_back(i);

    /**********/
    clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = SurCardGroupData;
    return;
}

void Put_All_SurCards( HandCardData &clsHandCardData, CardGroupData SurCardGroupData)
{

    /**/
    for (int i = 0; i < 18; i++)
        for (int j = 0; j< clsHandCardData.value_aHandCardList[i]; j++)
            clsHandCardData.value_nPutCardList.push_back(i);
    /**********/
    clsHandCardData.uctPutCardType  = SurCardGroupData;
    return;
}

void HandCardData::ClearPutCardList()
{
    color_nPutCardList.clear();
    
    value_nPutCardList.clear();
    
    uctPutCardType.cgType = cgERROR;
    uctPutCardType.nCount = 0;
    uctPutCardType.nMaxCard = -1;
    uctPutCardType.nValue = 0;
    
    return;
}


/**/

int cmp(int a, int b) { return a > b ? 1 : 0; }


void HandCardData::SortAsList(vector <int> & arr )
{
    sort(arr.begin(), arr.end(), cmp);
    return;
}

void HandCardData::get_valueHandCardList()
{
    //
    value_nHandCardList.clear();
    memset(value_aHandCardList, 0,sizeof(value_aHandCardList));
    
    for (vector<int>::iterator iter = color_nHandCardList.begin(); iter != color_nHandCardList.end(); iter++)
    {
        value_nHandCardList.push_back((*iter / 4)+3);
        value_aHandCardList[(*iter / 4) + 3]++;
    }
    
}

void HandCardData::Init()
{
    //
    get_valueHandCardList();
    
    // 
    SortAsList(color_nHandCardList);
    SortAsList(value_nHandCardList);
    
    //
    nHandCardCount = (int)value_nHandCardList.size();
    
}

/*
cgERROR
 */

CardGroupData ins_SurCardsType(int arr[])
{
    
    int nCount = 0;
    for (int i = 3; i < 18; i++)
    {
        nCount += arr[i];
    }
    
    CardGroupData retCardGroupData;
    retCardGroupData.nCount = nCount;
    
    
    //
    if (nCount == 1)
    {
        //
        int prov = 0;
        int SumValue = 0;
        for (int i = 3; i < 18; i++)
        {
            if (arr[i] == 1)
            {
                SumValue = i - 10;
                prov++;
                retCardGroupData.nMaxCard = i;
            }
        }
        if (prov == 1)
        {
            retCardGroupData.cgType = cgSINGLE;
            retCardGroupData.nValue= SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 2)
    {
        //
        int prov = 0;
        int SumValue = 0;
        int i = 0;
        for (i = 3; i < 16; i++)
        {
            if (arr[i] == 2)
            {
                SumValue = i - 10;
                prov++;
                retCardGroupData.nMaxCard = i;
            }
        }
        if (prov == 1)
        {
            retCardGroupData.cgType = cgDOUBLE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 3)
    {
        //
        int prov = 0;
        int SumValue = 0;
        int i = 0;
        for (i = 3; i < 16; i++)
        {
            if (arr[i] == 3)
            {
                SumValue = i - 10;
                prov++;
                retCardGroupData.nMaxCard = i;
            }
        }
        if (prov == 1)
        {
            retCardGroupData.cgType = cgTHREE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 4)
    {
        //
        int prov1 = 0;
        int prov2 = 0;
        int SumValue = 0;
        for (int i = 3; i < 18; i++)
        {
            if (arr[i] == 3)
            {
                SumValue = i - 10;
                prov1++;
                retCardGroupData.nMaxCard = i;
                
            }
            if (arr[i] == 1)
            {
                prov2++;
            }
            
        }
        if (prov1 == 1 && prov2 == 1)
        {
            retCardGroupData.cgType = cgTHREE_TAKE_ONE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 5)
    {
        //
        int prov1 = 0;
        int prov2 = 0;
        int SumValue = 0;
        for (int i = 3; i < 16; i++)
        {
            if (arr[i] == 3)
            {
                SumValue = i - 10;
                prov1++;
                retCardGroupData.nMaxCard = i;
                
            }
            if (arr[i] == 2)
            {
                prov2++;
                
            }
        }
        if (prov1 == 1 && prov2 == 1)
        {
            retCardGroupData.cgType = cgTHREE_TAKE_TWO;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 6)
    {
        //
        int prov1 = 0;
        int prov2 = 0;
        int SumValue = 0;
        for (int i = 3; i < 18; i++)
        {
            if (arr[i] == 4)
            {
                SumValue = (i - 3) / 2;
                prov1++;
                retCardGroupData.nMaxCard = i;
                
            }
            if (arr[i] == 1|| arr[i] == 2)
            {
                prov2+= arr[i];
            }
        }
        
        if (prov1 == 1 && prov2 == 2)
        {
            retCardGroupData.cgType = cgFOUR_TAKE_ONE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 8)
    {
        //
        int prov1 = 0;
        int prov2 = 0;
        int SumValue = 0;
        for (int i = 3; i < 16; i++)
        {
            if (arr[i] == 4)
            {
                SumValue = (i - 3) / 2;
                
                prov1++;
                retCardGroupData.nMaxCard = i;
            }
            if (arr[i] == 2|| arr[i] == 4)
            {
                prov2+= arr[i]/2;
                
            }
        }
        //prov2==4
        if (prov1 == 1 && prov2 == 4)
        {
            retCardGroupData.cgType = cgFOUR_TAKE_TWO;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 4)
    {
        //
        int prov = 0;
        int SumValue = 0;
        for (int i = 3; i < 16; i++)
        {
            if (arr[i] == 4)
            {
                SumValue += i - 3 + 7;
                prov++;
                retCardGroupData.nMaxCard = i;
            }
        }
        if (prov == 1)
        {
            retCardGroupData.cgType = cgBOMB_CARD;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount == 2)
    {
        int SumValue = 0;
        if (arr[17] > 0 && arr[16] > 0)
        {
            SumValue = 20;
            retCardGroupData.nMaxCard = 17;
            retCardGroupData.cgType = cgKING_CARD;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount >= 5)
    {
        //
        int prov = 0;
        int SumValue = 0;
        int i;
        for (i = 3; i < 15; i++)
        {
            if (arr[i] == 1)
            {
                prov++;
            }
            else
            {
                if (prov != 0)
                {
                    break;
                }
                
            }
        }
        SumValue = i - 10;
        
        if (prov == nCount)
        {
            retCardGroupData.nMaxCard = i-1;
            retCardGroupData.cgType = cgSINGLE_LINE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount >= 6)
    {
        //
        int prov = 0;
        int SumValue = 0;
        int i;
        for (i = 3; i < 15; i++)
        {
            if (arr[i] == 2)
            {
                prov++;
            }
            else
            {
                if (prov != 0)
                {
                    break;
                }
                
            }
        }
        SumValue = i - 10;
        
        if (prov * 2 == nCount)
        {
            retCardGroupData.nMaxCard = i - 1;
            retCardGroupData.cgType = cgDOUBLE_LINE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount >= 6)
    {
        //
        int prov = 0;
        
        int SumValue = 0;
        int i;
        for (i = 3; i < 15; i++)
        {
            if (arr[i] == 3)
            {
                prov++;
            }
            else
            {
                if (prov != 0)
                {
                    break;
                }
                
            }
        }
        SumValue = (i - 3) / 2;
        
        if (prov * 3 == nCount)
        {
            retCardGroupData.nMaxCard = i - 1;
            retCardGroupData.cgType = cgTHREE_LINE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    //
    if (nCount >= 8)
    {
        //
        int prov1 = 0, prov2 = 0;
        int SumValue = 0;
        int i, j;
        for (i = 3; i < 15; i++)
        {
            if (arr[i] == 3)
            {
                prov1++;
            }
            else
            {
                if (prov1 != 0)
                {
                    break;
                }
                
            }
        }
        for (j = 3; j < 18; j++)
        {
            if (arr[j] == 1)
            {
                prov2++;
            }
        }
        SumValue = (i - 3)/2;
        if (prov1 == prov2 && prov1 * 4 == nCount)
        {
            retCardGroupData.nMaxCard = i - 1;
            retCardGroupData.cgType = cgTHREE_TAKE_ONE_LINE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
        
    }
    //
    if (nCount >= 10)
    {
        //
        int prov1 = 0;
        int prov2 = 0;
        int SumValue = 0;
        int i, j;
        for (i = 3; i < 15; i++)
        {
            if (arr[i] == 3)
            {
                prov1++;
            }
            else
            {
                if (prov1 != 0)
                {
                    break;
                }
            }
        }
        for (j = 3; j < 16; j++)
        {
            if (arr[j] == 2)
            {
                prov2++;
            }
        }
        SumValue = (i - 3) / 2;
        if (prov1 == prov2&&prov1 * 5 == nCount)
        {
            retCardGroupData.nMaxCard = i - 1;
            retCardGroupData.cgType = cgTHREE_TAKE_TWO_LINE;
            retCardGroupData.nValue = SumValue;
            return retCardGroupData;
        }
    }
    
    retCardGroupData.cgType = cgERROR;
    return retCardGroupData;
}

/*
 vector
 
   
 cgERROR
 */
CardGroupData ins_SurCardsType(vector<int>list)
{
    int arr[18];
    memset(arr, 0, sizeof(arr));
    for (vector<int>::iterator iter = list.begin(); iter != list.end(); iter++)  
    {  
        arr[*iter]++;  
    }  
    return ins_SurCardsType(arr);  
}  

void HandCardData::PrintAll()
{

    py::print("color_nHandCardList:");
    for (vector<int>::iterator iter = color_nHandCardList.begin(); iter != color_nHandCardList.end(); iter++)
        py::print(get_CardsName(*iter), "end"_a=(iter == color_nHandCardList.end() - 1 ? '\n' : ','));
    
    py::print("");
    /*
     cout << "value_nHandCardList" << endl;
     for (vector<int>::iterator iter = value_nHandCardList.begin(); iter != value_nHandCardList.end(); iter++)
     cout << *iter << (iter == value_nHandCardList.end() - 1 ? '\n' : ',');
     
     cout << endl;
     
     cout << "value_aHandCardList" << endl;
     for (int i = 0; i < 18; i++)
     {
     cout << value_aHandCardList[i] << (i == 17 ? '\n' : ',');
     }
     
     cout << endl;
     
     
     cout << "nHandCardCount:" << nHandCardCount << endl;
     
     cout << endl;
     
     cout << "nGameRole:" << nGameRole << endl;
     
     cout << endl; 
     */  
}

bool  HandCardData::PutCards()
{
    for (vector<int>::iterator iter = value_nPutCardList.begin(); iter != value_nPutCardList.end(); iter++)
    {
        int color_nCard = -1;
        if (PutOneCard(*iter, color_nCard))
        {
            color_nPutCardList.push_back(color_nCard);
        }
        else
        {
            return false;
        }
    }
    
    nHandCardCount -= (int)value_nPutCardList.size();
    return true;
}

bool  HandCardData::PutOneCard(int value_nCard, int &color_nCard)
{
    bool ret = false;
    
    
    
    //value
    
    value_aHandCardList[value_nCard]--;
    
    if (value_aHandCardList[value_nCard] < 0)
    {
        return false;
    }
    
    
    //value
    for (vector<int>::iterator iter = value_nHandCardList.begin(); iter != value_nHandCardList.end(); iter++)
    {
        if (*iter == value_nCard)
        {
            value_nHandCardList.erase(iter);
            ret = true;
            break;
        }
    }
    
    
    // color
    
    int k = (value_nCard - 3) * 4;      //
    
    for (vector<int>::iterator iter = color_nHandCardList.begin(); iter != color_nHandCardList.end(); iter++)
    {
        
        for (int i = k; i < k + 4; i++)
        {
            if (*iter == i)
            {
                color_nCard = i;
                color_nHandCardList.erase(iter);
                return ret;
                
            }
        }
    }
    return false;  
}


/*
 
 CardGroupType cgType
 int MaxCard
 int Count
 
 CardGroupData
 */

CardGroupData get_GroupData(CardGroupType cgType, int MaxCard, int Count)
{
    CardGroupData uctCardGroupData;
    
    uctCardGroupData.cgType = cgType;
    uctCardGroupData.nCount = Count;
    uctCardGroupData.nMaxCard = MaxCard;
    
    //
    if (cgType == cgZERO)
        uctCardGroupData.nValue = 0;
    //
    else if (cgType == cgSINGLE)
        uctCardGroupData.nValue = MaxCard - 10;
    //
    else if (cgType == cgDOUBLE)
        uctCardGroupData.nValue = MaxCard - 10;
    //
    else if (cgType == cgTHREE)
        uctCardGroupData.nValue = MaxCard - 10;
    //
    else if (cgType == cgSINGLE_LINE)
        uctCardGroupData.nValue = MaxCard - 10 + 1;
    //
    else if (cgType == cgDOUBLE_LINE)
        uctCardGroupData.nValue = MaxCard - 10 + 1;
    //
    else if (cgType == cgTHREE_LINE)
        uctCardGroupData.nValue = (MaxCard - 3 + 1)/2;
    //
    else if (cgType == cgTHREE_TAKE_ONE)
        uctCardGroupData.nValue = MaxCard - 10;
    //
    else if (cgType == cgTHREE_TAKE_TWO)
        uctCardGroupData.nValue = MaxCard - 10;
    //
    else if (cgType == cgTHREE_TAKE_ONE_LINE)
        uctCardGroupData.nValue = (MaxCard - 3 + 1) / 2;
    //
    else if (cgType == cgTHREE_TAKE_TWO_LINE)
        uctCardGroupData.nValue = (MaxCard - 3 + 1) / 2;
    //
    else if (cgType == cgFOUR_TAKE_ONE)
        uctCardGroupData.nValue = (MaxCard - 3 ) / 2;
    //
    else if (cgType == cgFOUR_TAKE_TWO)
        uctCardGroupData.nValue = (MaxCard - 3 ) / 2;
    //
    else if (cgType == cgBOMB_CARD)
        uctCardGroupData.nValue = MaxCard - 3 + 7;
    //
    else if (cgType == cgKING_CARD)
        uctCardGroupData.nValue = 20;
    //
    else
        uctCardGroupData.nValue = 0;
    
    
    return uctCardGroupData;
}
/*
compute the weight of a given group
*/

void get_PutCardList_2_limit(GameSituation &clsGameSituation, HandCardData &clsHandCardData)
{
    clsHandCardData.ClearPutCardList();
    // aHandCardList: one hot representation of hand cards
    
    /**/
    if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
    {
        
        clsHandCardData.value_aHandCardList[17] --;
        clsHandCardData.value_aHandCardList[16] --;
        clsHandCardData.nHandCardCount -= 2;
        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
        clsHandCardData.value_aHandCardList[16] ++;
        clsHandCardData.value_aHandCardList[17] ++;
        clsHandCardData.nHandCardCount += 2;
        if (tmpHandCardValue.NeedRound == 1)
        {
            clsHandCardData.value_nPutCardList.push_back(17);
            clsHandCardData.value_nPutCardList.push_back(16);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
            return;
        }
    }
    
    
    //  
    if (clsGameSituation.uctNowCardGroup.cgType == cgERROR)
    {
        clsHandCardData.uctPutCardType = get_GroupData(cgERROR, 0, 0);
        return;
    }
    // 
    else if (clsGameSituation.uctNowCardGroup.cgType == cgZERO)
    {
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgSINGLE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgSINGLE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD|| SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard=0;
        //
        bool PutCards = false;
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 18; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 0)
            {
                //
                clsHandCardData.value_aHandCardList[i]--;
                clsHandCardData.nHandCardCount--;
                HandCardValue tmpHandCardValue=get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i]++;
                clsHandCardData.nHandCardCount++;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue-(BestHandCardValue.NeedRound*7)) <= (tmpHandCardValue.SumValue-(tmpHandCardValue.NeedRound*7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgSINGLE, BestMaxCard, 1);  
            return;  
        }
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }  
        }
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgDOUBLE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgDOUBLE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        //--------------------------------------------------------------------------------------
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 18; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 1)
            {
                //
                clsHandCardData.value_aHandCardList[i]-=2;
                clsHandCardData.nHandCardCount-=2;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i]+=2;
                clsHandCardData.nHandCardCount+=2;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgDOUBLE, BestMaxCard, 2);
            return;
        }
        
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] ==4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }  
        
        //  
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);  
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        //--------------------------------------------------------------------------------------
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 18; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                //
                clsHandCardData.value_aHandCardList[i] -= 3;
                clsHandCardData.nHandCardCount -= 3;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 3;
                clsHandCardData.nHandCardCount += 3;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE, BestMaxCard, 3);
            return;
        }
        
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }  
        
        //  
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);  
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgSINGLE_LINE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType == cgSINGLE_LINE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
            &&SurCardGroupData.nCount== clsGameSituation.uctNowCardGroup.nCount)
        {
            Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
            return;
        }
        //
        int prov = 0;
        //
        int start_i = 0;
        //
        int end_i = 0;
        //
        int length = clsGameSituation.uctNowCardGroup.nCount;
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;

        // if (length == 5) {
        //     std::cout << "last nMaxCard " << clsGameSituation.uctNowCardGroup.nMaxCard << std::endl;
        // }
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard - length + 2; i < 15; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 0)
            {
                prov++;
            }
            else
            {
                prov = 0;
            }
            if (prov >= length)
            {
                end_i = i;
                start_i = i - length + 1;
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] --;
                }
                clsHandCardData.nHandCardCount -= clsGameSituation.uctNowCardGroup.nCount;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] ++;
                }
                clsHandCardData.nHandCardCount += clsGameSituation.uctNowCardGroup.nCount;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {  
                    BestHandCardValue = tmpHandCardValue;  
                    BestMaxCard = end_i;  
                    PutCards = true;  
                }  
                
            }  
        }
        
        if (PutCards)
        {
            if (BestMaxCard <= clsGameSituation.uctNowCardGroup.nMaxCard) {
                printf("WARNING from C++...\n");
            }
            for (int j = BestMaxCard - length + 1; j <= BestMaxCard; j++)
            {
                // std::cout << j << ",";
                clsHandCardData.value_nPutCardList.push_back(j);
            }
            // std::cout << std::endl;
            // std::cout << "Setting nMaxCard " << BestMaxCard << std::endl;
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgSINGLE_LINE, BestMaxCard, clsGameSituation.uctNowCardGroup.nCount);
            return;  
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }

        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgDOUBLE_LINE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgDOUBLE_LINE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        //
        int prov = 0;
        //
        int start_i = 0;
        //
        int end_i = 0;
        //
        int length = clsGameSituation.uctNowCardGroup.nCount/2;
        //2+1
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard - length + 2; i < 15; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 1)
            {
                prov++;
            }
            else
            {
                prov = 0;
            }
            if (prov >= length)
            {
                end_i = i;
                start_i = i - length + 1;
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] -=2;
                }
                clsHandCardData.nHandCardCount -= clsGameSituation.uctNowCardGroup.nCount;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] +=2;
                }
                clsHandCardData.nHandCardCount += clsGameSituation.uctNowCardGroup.nCount;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = end_i;
                    PutCards = true;
                }
                
            }
        }
        
        if (PutCards)
        {
            for (int j = BestMaxCard - length + 1; j <= BestMaxCard; j++)
            {
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
            }
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgDOUBLE_LINE, BestMaxCard, clsGameSituation.uctNowCardGroup.nCount);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }
        
        
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
        
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE_LINE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE_LINE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        //
        int prov = 0;
        //
        int start_i = 0;
        //
        int end_i = 0;
        //
        int length = clsGameSituation.uctNowCardGroup.nCount / 3;
        //2+1
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard - length + 2; i < 15; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                prov++;
            }
            else
            {
                prov = 0;
            }
            if (prov >= length)
            {
                end_i = i;
                start_i = i - length + 1;
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] -= 3;
                }
                clsHandCardData.nHandCardCount -= clsGameSituation.uctNowCardGroup.nCount;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] += 3;
                }
                clsHandCardData.nHandCardCount += clsGameSituation.uctNowCardGroup.nCount;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = end_i;
                    PutCards = true;
                }
                
            }
        }
        
        if (PutCards)
        {
            for (int j = BestMaxCard - length + 1; j <= BestMaxCard; j++)
            {
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
            }
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE_LINE, BestMaxCard, clsGameSituation.uctNowCardGroup.nCount);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);  
                return;  
            }  
        }  
        
        
        
        //  
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);  
        return;  
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE_TAKE_ONE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE_TAKE_ONE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        //7
        BestHandCardValue.NeedRound += 1;
        //
        int BestMaxCard = 0;
        //
        int tmp_1 = 0;
        //
        bool PutCards = false;
        //
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] >2)
            {
                for (int j = 3; j < 18; j++)
                {
                    //
                    if (clsHandCardData.value_aHandCardList[j] > 0 && j != i)
                    {
                        clsHandCardData.value_aHandCardList[i] -= 3;
                        clsHandCardData.value_aHandCardList[j] -= 1;
                        clsHandCardData.nHandCardCount -= 4;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        clsHandCardData.value_aHandCardList[i] += 3;
                        clsHandCardData.value_aHandCardList[j] += 1;
                        clsHandCardData.nHandCardCount += 4;
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestMaxCard = i;
                            tmp_1 = j;
                            PutCards = true;
                        }
                    }
                }
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);  
            clsHandCardData.value_nPutCardList.push_back(tmp_1);  
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE_TAKE_ONE, BestMaxCard, 4);  
            return;  
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }
        
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE_TAKE_TWO)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE_TAKE_TWO&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        //7
        BestHandCardValue.NeedRound += 1;
        //
        int BestMaxCard = 0;
        //
        int tmp_1 = 0;
        //
        bool PutCards = false;
        
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] >2)
            {
                for (int j = 3; j < 16; j++)
                {
                    //
                    if (clsHandCardData.value_aHandCardList[j] > 1 && j != i)
                    {
                        clsHandCardData.value_aHandCardList[i] -= 3;
                        clsHandCardData.value_aHandCardList[j] -= 2;
                        clsHandCardData.nHandCardCount -= 5;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        clsHandCardData.value_aHandCardList[i] += 3;
                        clsHandCardData.value_aHandCardList[j] += 2;
                        clsHandCardData.nHandCardCount += 5;
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestMaxCard = i;
                            tmp_1 = j;
                            PutCards = true;
                        }
                    }
                }
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE_TAKE_TWO, BestMaxCard, 5);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE_TAKE_ONE_LINE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE_TAKE_ONE_LINE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }

        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        //
        int prov = 0;
        //
        int start_i = 0;
        //
        int end_i = 0;
        //
        int length = clsGameSituation.uctNowCardGroup.nCount / 4;
        
        int tmp_1 = 0;
        int tmp_2 = 0;
        int tmp_3 = 0;
        int tmp_4 = 0;
        //2+1
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard - length + 2; i < 15; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 3)
            {
                prov++;
            }
            else
            {
                prov = 0;
            }
            if (prov >= length)
            {
                end_i = i;
                start_i = i - length + 1;
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] -= 3;
                }
                clsHandCardData.nHandCardCount -= clsGameSituation.uctNowCardGroup.nCount;
                
                /*2-4
                 */
                //
                if (length == 2)
                {
                    // printf("start_i: %d, end_i: %d\n", start_i, end_i);
                    for (int j = 3; j < 18; j++)
                    {
                        if (j >= start_i && j <= end_i)
                        {
                            continue;
                        }
                        if (clsHandCardData.value_aHandCardList[j] > 0)
                        {
                            clsHandCardData.value_aHandCardList[j] -= 1;
                            for (int k = 3; k < 18; k++)
                            {
                                if (k >= start_i && k <= end_i)
                                {
                                    continue;
                                }
                                if (clsHandCardData.value_aHandCardList[k] > 0 && k != j)
                                {
                                    // printf("j, k: %d, %d\n", j, k);
                                    clsHandCardData.value_aHandCardList[k] -= 1;
                                    HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                    clsHandCardData.value_aHandCardList[k] += 1;
                                    
                                    //-*7  n -7
                                    if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                    {
                                        BestHandCardValue = tmpHandCardValue;
                                        BestMaxCard = end_i;
                                        tmp_1 = j;
                                        tmp_2 = k;
                                        PutCards = true;
                                    }
                                }
                            }
                            clsHandCardData.value_aHandCardList[j] += 1;
                        }
                        
                    }
                }
                //
                if (length == 3)
                {
                    for (int j = 3; j < 18; j++)
                    {
                        if (j >= start_i && j <= end_i)
                        {
                            continue;
                        }
                        if (clsHandCardData.value_aHandCardList[j] > 0)
                        {
                            clsHandCardData.value_aHandCardList[j] -= 1;
                            for (int k = 3; k < 18; k++)
                            {
                                if (clsHandCardData.value_aHandCardList[k] > 0 && k != j)
                                {
                                    if (k >= start_i && k <= end_i)
                                    {
                                        continue;
                                    }
                                    clsHandCardData.value_aHandCardList[k] -= 1;
                                    for (int l = 3; l < 18; l++)
                                    {
                                        if (l >= start_i && l <= end_i)
                                        {
                                            continue;
                                        }
                                        if (clsHandCardData.value_aHandCardList[l] > 0 && l != k && l != j)
                                        {
                                            clsHandCardData.value_aHandCardList[l] -= 1;
                                            HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                            //-*7  n -7
                                            if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                            {
                                                BestHandCardValue = tmpHandCardValue;
                                                BestMaxCard = end_i;
                                                tmp_1 = j;
                                                tmp_2 = k;
                                                tmp_3 = l;
                                                PutCards = true;
                                            }
                                            clsHandCardData.value_aHandCardList[l] += 1;
                                        }
                                    }
                                    clsHandCardData.value_aHandCardList[k] += 1;
                                }
                            }
                            clsHandCardData.value_aHandCardList[j] += 1;
                        }
                        
                        
                    }
                }
                //
                if (length == 4)
                {
                    for (int j = 3; j < 18; j++)
                    {
                        if (j >= start_i && j <= end_i)
                        {
                            continue;
                        }
                        if (clsHandCardData.value_aHandCardList[j] > 0)
                        {
                            clsHandCardData.value_aHandCardList[j] -= 1;
                            for (int k = 3; k < 18; k++)
                            {
                                if (k >= start_i && k <= end_i)
                                {
                                    continue;
                                }
                                if (clsHandCardData.value_aHandCardList[k] > 0 &&  k != j)
                                {
                                    clsHandCardData.value_aHandCardList[k] -= 1;
                                    for (int l = 3; l < 18; l++)
                                    {
                                        if (l >= start_i && l <= end_i)
                                        {
                                            continue;
                                        }
                                        if (clsHandCardData.value_aHandCardList[l] > 0 && l != k && l != j)
                                        {
                                            clsHandCardData.value_aHandCardList[l] -= 1;
                                            for (int m = 3; m < 18; m++)
                                            {
                                                if (m >= start_i && m <= end_i)
                                                {
                                                    continue;
                                                }
                                                if (clsHandCardData.value_aHandCardList[m] > 0 && m != l && m != k && m != j)
                                                {
                                                    clsHandCardData.value_aHandCardList[m] -= 1;
                                                    HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                                    //-*7  n -7
                                                    if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                                    {
                                                        BestHandCardValue = tmpHandCardValue;
                                                        BestMaxCard = end_i;
                                                        tmp_1 = j;
                                                        tmp_2 = k;
                                                        tmp_3 = l;
                                                        tmp_4 = m;
                                                        PutCards = true;
                                                    }
                                                    clsHandCardData.value_aHandCardList[m] += 1;
                                                }
                                            }
                                            clsHandCardData.value_aHandCardList[l] += 1;
                                        }
                                    }
                                    clsHandCardData.value_aHandCardList[k] += 1;
                                }
                            }
                            clsHandCardData.value_aHandCardList[j] += 1;
                        }
                        
                        
                    }
                }
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] += 3;
                }
                clsHandCardData.nHandCardCount += clsGameSituation.uctNowCardGroup.nCount;
            }
        }
        
        if (PutCards)
        {
            for (int j = BestMaxCard - length + 1; j <= BestMaxCard; j++)
            {
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
            }
            
            if (length == 2)
            {
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_2);  
            }  
            if (length == 3)  
            {  
                clsHandCardData.value_nPutCardList.push_back(tmp_1);  
                clsHandCardData.value_nPutCardList.push_back(tmp_2);  
                clsHandCardData.value_nPutCardList.push_back(tmp_3);  
                
            }  
            if (length == 4)  
            {  
                clsHandCardData.value_nPutCardList.push_back(tmp_1);  
                clsHandCardData.value_nPutCardList.push_back(tmp_2);  
                clsHandCardData.value_nPutCardList.push_back(tmp_3);  
                clsHandCardData.value_nPutCardList.push_back(tmp_4);  
            }  
            
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE_TAKE_ONE_LINE, BestMaxCard, clsGameSituation.uctNowCardGroup.nCount);  
            return;  
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }

        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgTHREE_TAKE_TWO_LINE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgTHREE_TAKE_TWO_LINE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        
        
        //7
        BestHandCardValue.NeedRound += 1;
        
        //
        int BestMaxCard = 0;
        //
        bool PutCards = false;
        //
        int prov = 0;
        //
        int start_i = 0;
        //
        int end_i = 0;
        //
        int length = clsGameSituation.uctNowCardGroup.nCount / 4;
        
        int tmp_1 = 0;
        int tmp_2 = 0;
        int tmp_3 = 0;
        //2+1
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard - length + 2; i < 15; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                prov++;
            }
            else
            {
                prov = 0;
            }
            if (prov >= length)
            {
                end_i = i;
                start_i = i - length + 1;
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] -= 3;
                }
                clsHandCardData.nHandCardCount -= clsGameSituation.uctNowCardGroup.nCount;
                
                /*2-4
                 */
                //
                if (length == 2)
                {
                    for (int j = 3; j < 18; j++)
                    {
                        if (clsHandCardData.value_aHandCardList[j] > 1)
                        {
                            clsHandCardData.value_aHandCardList[j] -= 2;
                            for (int k = 3; k < 18; k++)
                            {
                                if (clsHandCardData.value_aHandCardList[k] > 1 &&  k != j)
                                {
                                    clsHandCardData.value_aHandCardList[k] -= 2;
                                    HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                    clsHandCardData.value_aHandCardList[k] += 2;
                                    
                                    //-*7  n -7
                                    if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                    {
                                        BestHandCardValue = tmpHandCardValue;
                                        BestMaxCard = end_i;
                                        tmp_1 = j;
                                        tmp_2 = k;
                                        PutCards = true;
                                    }
                                }
                            }
                            clsHandCardData.value_aHandCardList[j] += 2;
                        }
                        
                    }
                }
                //
                if (length == 3)
                {
                    for (int j = 3; j < 18; j++)
                    {
                        if (clsHandCardData.value_aHandCardList[j] > 1)
                        {
                            clsHandCardData.value_aHandCardList[j] -= 2;
                            for (int k = 3; k < 18; k++)
                            {
                                if (clsHandCardData.value_aHandCardList[k] > 1 &&  k != j)
                                {
                                    clsHandCardData.value_aHandCardList[k] -= 2;
                                    for (int l = 3; l < 18; l++)
                                    {
                                        if (clsHandCardData.value_aHandCardList[l] > 1 && l != k && l != j)
                                        {
                                            clsHandCardData.value_aHandCardList[l] -= 2;
                                            HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                            //-*7  n -7
                                            if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                            {
                                                BestHandCardValue = tmpHandCardValue;
                                                BestMaxCard = end_i;
                                                tmp_1 = j;
                                                tmp_2 = k;
                                                tmp_3 = l;
                                                PutCards = true;
                                            }
                                            clsHandCardData.value_aHandCardList[l] += 2;
                                        }
                                    }
                                    clsHandCardData.value_aHandCardList[k] += 2;
                                }
                            }
                            clsHandCardData.value_aHandCardList[j] += 2;
                        }
                        
                        
                    }
                }
                
                for (int j = start_i; j <= end_i; j++)
                {
                    clsHandCardData.value_aHandCardList[j] += 3;
                }
                clsHandCardData.nHandCardCount += clsGameSituation.uctNowCardGroup.nCount;
            }
        }
        
        if (PutCards)
        {
            for (int j = BestMaxCard - length + 1; j <= BestMaxCard; j++)
            {
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
                clsHandCardData.value_nPutCardList.push_back(j);
            }
            
            if (length == 2)
            {
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_2);
                clsHandCardData.value_nPutCardList.push_back(tmp_2);
            }
            if (length == 3)
            {
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_2);
                clsHandCardData.value_nPutCardList.push_back(tmp_2);
                clsHandCardData.value_nPutCardList.push_back(tmp_3);
                clsHandCardData.value_nPutCardList.push_back(tmp_3);
                
            }
            
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgTHREE_TAKE_TWO_LINE, BestMaxCard, clsGameSituation.uctNowCardGroup.nCount);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;

    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgFOUR_TAKE_ONE)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgFOUR_TAKE_ONE&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        //7
        BestHandCardValue.NeedRound += 1;
        //
        int BestMaxCard = 0;
        //
        int tmp_1 = 0, tmp_2 = 0;
        //
        bool PutCards = false;
        
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                for (int j = 3; j < 18; j++)
                {
                    //
                    if (clsHandCardData.value_aHandCardList[j] > 0 && j != i)
                    {
                        //,
                        for (int k = j + 1; k < 18; k++)
                        {
                            if (clsHandCardData.value_aHandCardList[k] > 0 && k != i)
                            {
                                clsHandCardData.value_aHandCardList[i] -= 4;
                                clsHandCardData.value_aHandCardList[j] -= 1;
                                clsHandCardData.value_aHandCardList[k] -= 1;
                                clsHandCardData.nHandCardCount -= 6;
                                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                clsHandCardData.value_aHandCardList[i] += 4;
                                clsHandCardData.value_aHandCardList[j] += 1;
                                clsHandCardData.value_aHandCardList[k] += 1;
                                clsHandCardData.nHandCardCount += 6;
                                
                                //-*7  n -7
                                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                {
                                    BestHandCardValue = tmpHandCardValue;
                                    BestMaxCard = i;
                                    tmp_1 = j;
                                    tmp_2 = k;
                                    PutCards = true;
                                }
                            }
                        }
                    }
                }
            }  
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgFOUR_TAKE_ONE, BestMaxCard, 6);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                
                //,
                clsHandCardData.value_aHandCardList[i] -= 4;
                clsHandCardData.nHandCardCount -= 4;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 4;
                clsHandCardData.nHandCardCount += 4;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7))
                    // 
                    || tmpHandCardValue.SumValue > 0)
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestMaxCard = i;
                    PutCards = true;
                }
                
            }
        }
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, BestMaxCard, 4);
            return;
        }
        
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            //20
            if (BestHandCardValue.SumValue > 20)
            {
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                return;
            }
        }
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;

    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgFOUR_TAKE_TWO)
    {
        //
        CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
        if (SurCardGroupData.cgType != cgERROR)
        {
            if (SurCardGroupData.cgType == cgFOUR_TAKE_TWO&&SurCardGroupData.nMaxCard>clsGameSituation.uctNowCardGroup.nMaxCard
                &&SurCardGroupData.nCount == clsGameSituation.uctNowCardGroup.nCount)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
            else if (SurCardGroupData.cgType == cgBOMB_CARD || SurCardGroupData.cgType == cgKING_CARD)
            {
                Put_All_SurCards(clsGameSituation, clsHandCardData, SurCardGroupData);
                return;
            }
        }
        
        //
        HandCardValue BestHandCardValue = get_HandCardValue(clsHandCardData);
        //7
        BestHandCardValue.NeedRound += 1;
        //
        int BestMaxCard = 0;
        //
        int tmp_1 = 0, tmp_2 = 0;
        //
        bool PutCards = false;
        
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                for (int j = 3; j < 18; j++)
                {
                    //
                    if (clsHandCardData.value_aHandCardList[j] > 1 && j != i)
                    {
                        //,
                        for (int k = j + 1; k < 18; k++)
                        {
                            if (clsHandCardData.value_aHandCardList[k] > 1 && k != i)
                            {
                                clsHandCardData.value_aHandCardList[i] -= 4;
                                clsHandCardData.value_aHandCardList[j] -= 2;
                                clsHandCardData.value_aHandCardList[k] -= 2;
                                clsHandCardData.nHandCardCount -= 8;
                                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                clsHandCardData.value_aHandCardList[i] += 4;
                                clsHandCardData.value_aHandCardList[j] += 2;
                                clsHandCardData.value_aHandCardList[k] += 2;
                                clsHandCardData.nHandCardCount += 8;
                                
                                //-*7  n -7
                                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                {
                                    BestHandCardValue = tmpHandCardValue;
                                    BestMaxCard = i;
                                    tmp_1 = j;
                                    tmp_2 = k;
                                    PutCards = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        /*14
         14*/
        
        if (BestHandCardValue.SumValue > 14)
        {
            //
            for (int i = 3; i < 16; i++)
            {
                if (clsHandCardData.value_aHandCardList[i] == 4)
                {
                    clsHandCardData.value_nPutCardList.push_back(i);
                    clsHandCardData.value_nPutCardList.push_back(i);
                    clsHandCardData.value_nPutCardList.push_back(i);
                    clsHandCardData.value_nPutCardList.push_back(i);
                    
                    clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, i, 4);
                    
                    return;
                }
            }
            //
            if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
            {
                
                clsHandCardData.value_nPutCardList.push_back(17);
                clsHandCardData.value_nPutCardList.push_back(16);
                
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
                
                return;  
            }  
        }
        
        if (PutCards)
        {
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(BestMaxCard);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgFOUR_TAKE_TWO, BestMaxCard, 8);
            return;
        }
        
        //--------------------------------------------------------------------------------------
        
        //
        for (int i = 3; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, i, 4);
                
                return;
            }
        }
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            
            clsHandCardData.value_nPutCardList.push_back(17);
            clsHandCardData.value_nPutCardList.push_back(16);
            
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
            
            return;  
        }
        
        //
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    //
    else if (clsGameSituation.uctNowCardGroup.cgType == cgBOMB_CARD)
    {
        //
        for (int i = clsGameSituation.uctNowCardGroup.nMaxCard + 1; i < 16; i++)
        {
            if (clsHandCardData.value_aHandCardList[i] == 4)
            {
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                clsHandCardData.value_nPutCardList.push_back(i);
                
                clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgBOMB_CARD, i, 4);
                
                return;
            }
        }
        //
        if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
        {
            
            clsHandCardData.value_nPutCardList.push_back(17);
            clsHandCardData.value_nPutCardList.push_back(16);
            
            clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(cgKING_CARD, 17, 2);
            
            return;  
        }  
        //  
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);  
        return;
    }
    // 
    else if (clsGameSituation.uctNowCardGroup.cgType == cgKING_CARD)
    {
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
        return;
    }
    // 
    else
    {
        clsHandCardData.uctPutCardType = get_GroupData(cgZERO, 0, 0);
    }  
    return;  
}
void get_PutCardList_2_unlimit(HandCardData &clsHandCardData)
{
    
    clsHandCardData.ClearPutCardList();
    
    ///
    CardGroupData SurCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList); // get hand card information
    //
    if (SurCardGroupData.cgType != cgERROR&&SurCardGroupData.cgType!=cgFOUR_TAKE_ONE&&SurCardGroupData.cgType !=cgFOUR_TAKE_TWO)
    {
        Put_All_SurCards(clsHandCardData, SurCardGroupData);
        return;
    }
    
    /**/
    if (clsHandCardData.value_aHandCardList[17] > 0 && clsHandCardData.value_aHandCardList[16] > 0)
    {
        
        clsHandCardData.value_aHandCardList[17] --;
        clsHandCardData.value_aHandCardList[16] --;
        clsHandCardData.nHandCardCount -= 2;
        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
        clsHandCardData.value_aHandCardList[16] ++;
        clsHandCardData.value_aHandCardList[17] ++;
        clsHandCardData.nHandCardCount += 2;
        if (tmpHandCardValue.NeedRound == 1)
        {
            clsHandCardData.value_nPutCardList.push_back(17);
            clsHandCardData.value_nPutCardList.push_back(16);
            clsHandCardData.uctPutCardType = get_GroupData(cgKING_CARD, 17, 2);
            return;
        }
    }
    
    //
    HandCardValue BestHandCardValue;
    BestHandCardValue.NeedRound = 20;
    BestHandCardValue.SumValue = MinCardsValue;
    //7
    BestHandCardValue.NeedRound += 1;
    
    //
    CardGroupData BestCardGroup;
    
    //
    int tmp_1 = 0;
    int tmp_2 = 0;
    int tmp_3 = 0;
    int tmp_4 = 0;

    //
    for (int i = 3; i < 16; i++)
    {
        //2.0
        if (clsHandCardData.value_aHandCardList[i] != 4)
        {
            // 3 plus 1
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                clsHandCardData.value_aHandCardList[i] -= 3;
                for (int j = 3; j < 18; j++)
                {
                    if (clsHandCardData.value_aHandCardList[j] > 0)
                    {
                        clsHandCardData.value_aHandCardList[j] -= 1;
                        clsHandCardData.nHandCardCount -= 4;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        clsHandCardData.value_aHandCardList[j] += 1;
                        clsHandCardData.nHandCardCount += 4;
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestCardGroup = get_GroupData(cgTHREE_TAKE_ONE, i, 4);
                            tmp_1 = j;
                        }
                    }
                }
                clsHandCardData.value_aHandCardList[i] += 3;
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                for (int j = 3; j < 16; j++)
                {
                    clsHandCardData.value_aHandCardList[i] -= 3;
                    if (clsHandCardData.value_aHandCardList[j] > 1)
                    {
                        clsHandCardData.value_aHandCardList[j] -= 2;
                        clsHandCardData.nHandCardCount -= 5;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        clsHandCardData.value_aHandCardList[j] += 2;
                        clsHandCardData.nHandCardCount += 5;
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestCardGroup = get_GroupData(cgTHREE_TAKE_TWO, i, 5);
                            tmp_1 = j;
                        }
                    }
                    clsHandCardData.value_aHandCardList[i] += 3;
                }
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 3)
            {
                //2.0
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 3)
            {
                //2.0
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                int prov = 0;
                for (int j = i; j < 15; j++)
                {
                    if (clsHandCardData.value_aHandCardList[j] > 2)
                    {
                        prov++;
                    }
                    else
                    {
                        break;
                    }
                    /*2-4
                     */
                    //
                    if (prov == 2)
                    {
                        
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 4;
                        for (int tmp1 = 3; tmp1 < 18; tmp1++)
                        {
                            if (tmp1 >= i && tmp1 <= j)
                            {
                                continue;
                            }
                            if (clsHandCardData.value_aHandCardList[tmp1] > 0)
                            {
                                clsHandCardData.value_aHandCardList[tmp1] -= 1;
                                for (int tmp2 = tmp1 + 1; tmp2 < 18; tmp2++)
                                {
                                    if (tmp2 >= i && tmp2 <= j)
                                    {
                                        continue;
                                    }
                                    if (clsHandCardData.value_aHandCardList[tmp2] > 0)
                                    {
                                        clsHandCardData.value_aHandCardList[tmp2] -= 1;
                                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                        
                                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                        {
                                            BestHandCardValue = tmpHandCardValue;
                                            BestCardGroup = get_GroupData(cgTHREE_TAKE_ONE_LINE, j, prov * 4);
                                            tmp_1 = tmp1;
                                            tmp_2 = tmp2;
                                        }
                                        clsHandCardData.value_aHandCardList[tmp2] += 1;
                                    }
                                }
                                clsHandCardData.value_aHandCardList[tmp1] += 1;
                            }
                        }
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] += 3;
                        }
                        clsHandCardData.nHandCardCount += prov * 4;
                    }
                    //
                    if (prov == 3)
                    {
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 4;
                        for (int tmp1 = 3; tmp1 < 18; tmp1++)
                        {
                            if (tmp1 >= i && tmp1 <= j)
                            {
                                continue;
                            }
                            if (clsHandCardData.value_aHandCardList[tmp1] > 0)
                            {
                                clsHandCardData.value_aHandCardList[tmp1] -= 1;
                                for (int tmp2 = tmp1 + 1; tmp2 < 18; tmp2++)
                                {
                                    if (tmp2 >= i && tmp2 <= j)
                                    {
                                        continue;
                                    }
                                    if (clsHandCardData.value_aHandCardList[tmp2] > 0)
                                    {
                                        clsHandCardData.value_aHandCardList[tmp2] -= 1;
                                        for (int tmp3 = tmp2 + 1; tmp3 < 18; tmp3++)
                                        {
                                            if (tmp3 >= i && tmp3 <= j)
                                            {
                                                continue;
                                            }
                                            if (clsHandCardData.value_aHandCardList[tmp3] > 0)
                                            {
                                                clsHandCardData.value_aHandCardList[tmp3] -= 1;
                                                
                                                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                                {
                                                    
                                                    BestHandCardValue = tmpHandCardValue;
                                                    BestCardGroup = get_GroupData(cgTHREE_TAKE_ONE_LINE, j, prov * 4);
                                                    tmp_1 = tmp1;
                                                    tmp_2 = tmp2;
                                                    tmp_3 = tmp3;
                                                    
                                                }
                                                clsHandCardData.value_aHandCardList[tmp3] += 1;
                                            }
                                            
                                        }
                                        clsHandCardData.value_aHandCardList[tmp2] += 1;
                                    }
                                    
                                }
                                clsHandCardData.value_aHandCardList[tmp1] += 1;
                            }
                        }
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] += 3;
                        }
                        clsHandCardData.nHandCardCount += prov * 4;
                    }
                    //
                    if (prov == 4)
                    {
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 4;
                        for (int tmp1 = 3; tmp1 < 18; tmp1++)
                        {
                            if (tmp1 >= i && tmp1 <= j)
                            {
                                continue;
                            }
                            if (clsHandCardData.value_aHandCardList[tmp1] > 0)
                            {
                                clsHandCardData.value_aHandCardList[tmp1] -= 1;
                                for (int tmp2 = tmp1 + 1; tmp2 < 18; tmp2++)
                                {
                                    if (tmp2 >= i && tmp2 <= j)
                                    {
                                        continue;
                                    }
                                    if (clsHandCardData.value_aHandCardList[tmp2] > 0)
                                    {
                                        clsHandCardData.value_aHandCardList[tmp2] -= 1;
                                        for (int tmp3 = tmp2 + 1; tmp3 < 18; tmp3++)
                                        {
                                            if (tmp3 >= i && tmp3 <= j)
                                            {
                                                continue;
                                            }
                                            if (clsHandCardData.value_aHandCardList[tmp3] > 0)
                                            {
                                                clsHandCardData.value_aHandCardList[tmp3] -= 1;
                                                for (int tmp4 = tmp3 + 1; tmp4 < 18; tmp4++)
                                                {
                                                    if (tmp4 >= i && tmp4 <= j)
                                                    {
                                                        continue;
                                                    }
                                                    if (clsHandCardData.value_aHandCardList[tmp4] > 0)
                                                    {
                                                        clsHandCardData.value_aHandCardList[tmp4] -= 1;
                                                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                                        {
                                                            BestHandCardValue = tmpHandCardValue;
                                                            BestCardGroup = get_GroupData(cgTHREE_TAKE_ONE_LINE, j, prov * 4);
                                                            tmp_1 = tmp1;
                                                            tmp_2 = tmp2;
                                                            tmp_3 = tmp3;
                                                            tmp_4 = tmp4;
                                                        }
                                                        clsHandCardData.value_aHandCardList[tmp4] += 1;
                                                    }
                                                    
                                                }
                                                clsHandCardData.value_aHandCardList[tmp3] += 1;
                                            }
                                            
                                        }
                                        clsHandCardData.value_aHandCardList[tmp2] += 1;
                                    }
                                    
                                }
                                clsHandCardData.value_aHandCardList[tmp1] += 1;
                            }
                        }
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] += 3;
                        }
                        clsHandCardData.nHandCardCount += prov * 4;
                    }
                    //prov==5
                }
                
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                int prov = 0;
                for (int j = i; j < 15; j++)
                {
                    if (clsHandCardData.value_aHandCardList[j] > 2)
                    {
                        prov++;
                    }
                    else
                    {
                        break;
                    }
                    /*2-4
                     */
                    //
                    if (prov == 2)
                    {
                        
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 5;
                        for (int tmp1 = 3; tmp1 < 16; tmp1++)
                        {
                            if (tmp1 >= i && tmp1 <= j)
                            {
                                continue;
                            }
                            if (clsHandCardData.value_aHandCardList[tmp1] > 1)
                            {
                                clsHandCardData.value_aHandCardList[tmp1] -= 2;
                                for (int tmp2 = tmp1 + 1; tmp2 < 16; tmp2++)
                                {
                                    if (tmp2 >= i && tmp2 <= j)
                                    {
                                        continue;
                                    }
                                    if (clsHandCardData.value_aHandCardList[tmp2] > 1)
                                    {
                                        clsHandCardData.value_aHandCardList[tmp2] -= 2;
                                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                        {
                                            BestHandCardValue = tmpHandCardValue;
                                            BestCardGroup = get_GroupData(cgTHREE_TAKE_TWO_LINE, j, prov * 5);
                                            tmp_1 = tmp1;
                                            tmp_2 = tmp2;
                                        }
                                        clsHandCardData.value_aHandCardList[tmp2] += 2;
                                    }
                                }
                                clsHandCardData.value_aHandCardList[tmp1] += 2;
                            }
                        }
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] += 3;
                        }
                        clsHandCardData.nHandCardCount += prov * 5;
                    }
                    //
                    if (prov == 3)
                    {
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 5;
                        for (int tmp1 = 3; tmp1 < 16; tmp1++)
                        {
                            if (tmp1 >= i && tmp1 <= j)
                            {
                                continue;
                            }
                            if (clsHandCardData.value_aHandCardList[tmp1] > 1)
                            {
                                clsHandCardData.value_aHandCardList[tmp1] -= 2;
                                for (int tmp2 = tmp1 + 1; tmp2 < 16; tmp2++)
                                {
                                    if (tmp2 >= i && tmp2 <= j)
                                    {
                                        continue;
                                    }
                                    if (clsHandCardData.value_aHandCardList[tmp2] > 1)
                                    {
                                        clsHandCardData.value_aHandCardList[tmp2] -= 2;
                                        for (int tmp3 = tmp2 + 1; tmp3 < 16; tmp3++)
                                        {
                                            if (tmp3 >= i && tmp3 <= j)
                                            {
                                                continue;
                                            }
                                            if (clsHandCardData.value_aHandCardList[tmp3] > 1)
                                            {
                                                clsHandCardData.value_aHandCardList[tmp3] -= 2;
                                                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                                                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                                                {
                                                    BestHandCardValue = tmpHandCardValue;
                                                    BestCardGroup = get_GroupData(cgTHREE_TAKE_TWO_LINE, j, prov * 5);
                                                    tmp_1 = tmp1;
                                                    tmp_2 = tmp2;
                                                    tmp_3 = tmp3;
                                                }
                                                clsHandCardData.value_aHandCardList[tmp3] += 2;
                                            }
                                            
                                        }
                                        clsHandCardData.value_aHandCardList[tmp2] += 2;
                                    }
                                    
                                }
                                clsHandCardData.value_aHandCardList[tmp1] += 2;
                            }  
                        }  
                        for (int k = i; k <= j; k++)  
                        {  
                            clsHandCardData.value_aHandCardList[k] += 3;  
                        }  
                        clsHandCardData.nHandCardCount += prov * 5;  
                    }  
                    //prov==4  
                }  
            }  
        }  
        
    }
    //
    if (BestCardGroup.cgType == cgTHREE_TAKE_ONE)
    {
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(tmp_1);
        clsHandCardData.uctPutCardType = BestCardGroup;
        return;
    }
    else if (BestCardGroup.cgType == cgTHREE_TAKE_TWO)
    {
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
        clsHandCardData.value_nPutCardList.push_back(tmp_1);
        clsHandCardData.value_nPutCardList.push_back(tmp_1);
        clsHandCardData.uctPutCardType = BestCardGroup;
        return;
    }
    else if (BestCardGroup.cgType == cgTHREE_TAKE_ONE_LINE)
    {
        for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount / 4) + 1; j <= BestCardGroup.nMaxCard; j++)
        {
            clsHandCardData.value_nPutCardList.push_back(j);
            clsHandCardData.value_nPutCardList.push_back(j);
            clsHandCardData.value_nPutCardList.push_back(j);
        }
        
        if (BestCardGroup.nCount / 4 == 2)
        {
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
        }
        if (BestCardGroup.nCount / 4 == 3)
        {
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_3);
        }
        if (BestCardGroup.nCount / 4 == 4)
        {
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_3);
            clsHandCardData.value_nPutCardList.push_back(tmp_4);
        }
        
        clsHandCardData.uctPutCardType = BestCardGroup;
        return;
    }
    else if (BestCardGroup.cgType == cgTHREE_TAKE_TWO_LINE)
    {
        for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount / 5) + 1; j <= BestCardGroup.nMaxCard; j++)
        {
            clsHandCardData.value_nPutCardList.push_back(j);
            clsHandCardData.value_nPutCardList.push_back(j);
            clsHandCardData.value_nPutCardList.push_back(j);
        }
        if (BestCardGroup.nCount / 5 == 2)
        {
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
        }
        if (BestCardGroup.nCount / 5 == 3)
        {
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_1);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_2);
            clsHandCardData.value_nPutCardList.push_back(tmp_3);
            clsHandCardData.value_nPutCardList.push_back(tmp_3);
        }
        clsHandCardData.uctPutCardType = BestCardGroup;
        return;
    }
    
    //
    for (int i = 3; i < 16; i++)
    {
        if (clsHandCardData.value_aHandCardList[i] != 0 && clsHandCardData.value_aHandCardList[i] != 4) {
            //
            if (clsHandCardData.value_aHandCardList[i] > 0)
            {
                clsHandCardData.value_aHandCardList[i]--;
                clsHandCardData.nHandCardCount--;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i]++;
                clsHandCardData.nHandCardCount++;
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestCardGroup= get_GroupData(cgSINGLE, i, 1);
                }
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 1)
            {
                //
                clsHandCardData.value_aHandCardList[i] -= 2;
                clsHandCardData.nHandCardCount -= 2;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 2;
                clsHandCardData.nHandCardCount += 2;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestCardGroup = get_GroupData(cgDOUBLE, i, 2);
                }
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 2)
            {
                clsHandCardData.value_aHandCardList[i] -= 3;
                clsHandCardData.nHandCardCount -= 3;
                HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                clsHandCardData.value_aHandCardList[i] += 3;
                clsHandCardData.nHandCardCount += 3;
                
                //-*7  n -7
                if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                {
                    BestHandCardValue = tmpHandCardValue;
                    BestCardGroup = get_GroupData(cgTHREE, i, 3);
                }
            }
            
            //
            if (clsHandCardData.value_aHandCardList[i] > 0)
            {
                int prov = 0;
                for (int j = i; j < 15; j++)
                {
                    if(clsHandCardData.value_aHandCardList[j]>0)
                    {
                        prov++;
                    }
                    else
                    {
                        break;
                    }
                    if (prov >= 5)
                    {
                        
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] --;
                        }
                        clsHandCardData.nHandCardCount -= prov;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] ++;
                        }
                        clsHandCardData.nHandCardCount += prov;
                        
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestCardGroup = get_GroupData(cgSINGLE_LINE, j, prov);
                        }
                    }
                }
                
            }
            //
            if (clsHandCardData.value_aHandCardList[i] > 1)
            {
                int prov = 0;
                for (int j = i; j < 15; j++)
                {
                    if (clsHandCardData.value_aHandCardList[j]>1)
                    {
                        prov++;
                    }
                    else
                    {
                        break;
                    }
                    if (prov >= 3)
                    {
                        
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -=2;
                        }
                        clsHandCardData.nHandCardCount -= prov*2;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] +=2;
                        }
                        clsHandCardData.nHandCardCount += prov*2;
                        
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestCardGroup = get_GroupData(cgDOUBLE_LINE, j, prov*2);
                        }
                    }
                }
            }
            //
            if(clsHandCardData.value_aHandCardList[i] > 2)
            {
                int prov = 0;
                for (int j = i; j < 15; j++)
                {
                    if (clsHandCardData.value_aHandCardList[j]>2)
                    {
                        prov++;
                    }
                    else
                    {
                        break;
                    }
                    if (prov >= 2)
                    {
                        
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] -= 3;
                        }
                        clsHandCardData.nHandCardCount -= prov * 3;
                        HandCardValue tmpHandCardValue = get_HandCardValue(clsHandCardData);
                        for (int k = i; k <= j; k++)
                        {
                            clsHandCardData.value_aHandCardList[k] += 3;
                        }
                        clsHandCardData.nHandCardCount += prov * 3;
                        
                        //-*7  n -7
                        if ((BestHandCardValue.SumValue - (BestHandCardValue.NeedRound * 7)) <= (tmpHandCardValue.SumValue - (tmpHandCardValue.NeedRound * 7)))
                        {
                            BestHandCardValue = tmpHandCardValue;
                            BestCardGroup = get_GroupData(cgTHREE_LINE, j, prov * 3);
                        }
                    }
                }
            }
            if (BestCardGroup.cgType == cgERROR)
            {
                
            }
            else if (BestCardGroup.cgType == cgSINGLE)
            {
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgDOUBLE)
            {
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE)
            {
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgSINGLE_LINE)
            {
                for (int j = BestCardGroup.nMaxCard- BestCardGroup.nCount+1; j <= BestCardGroup.nMaxCard; j++)
                {
                    clsHandCardData.value_nPutCardList.push_back(j);
                }
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgDOUBLE_LINE)
            {
                for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount/2) + 1; j <= BestCardGroup.nMaxCard; j++)
                {
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                }
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE_LINE)
            {
                for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount / 3) + 1; j <= BestCardGroup.nMaxCard; j++)
                {
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                }
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE_TAKE_ONE)
            {
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE_TAKE_TWO)
            {
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(BestCardGroup.nMaxCard);
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.value_nPutCardList.push_back(tmp_1);
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE_TAKE_ONE_LINE)
            {
                for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount / 4) + 1; j <= BestCardGroup.nMaxCard; j++)
                {
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                }
                
                if (BestCardGroup.nCount / 4 == 2)
                {
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                }
                if (BestCardGroup.nCount / 4 == 3)
                {
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                    clsHandCardData.value_nPutCardList.push_back(tmp_3);
                }
                if (BestCardGroup.nCount / 4 == 4)
                {
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                    clsHandCardData.value_nPutCardList.push_back(tmp_3);
                    clsHandCardData.value_nPutCardList.push_back(tmp_4);
                }
                
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            else if (BestCardGroup.cgType == cgTHREE_TAKE_TWO_LINE)
            {
                for (int j = BestCardGroup.nMaxCard - (BestCardGroup.nCount / 5) + 1; j <= BestCardGroup.nMaxCard; j++)
                {
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                    clsHandCardData.value_nPutCardList.push_back(j);
                }
                if (BestCardGroup.nCount / 5 == 2)
                {
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                }
                if (BestCardGroup.nCount / 5 == 3)
                {
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_1);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                    clsHandCardData.value_nPutCardList.push_back(tmp_2);
                    clsHandCardData.value_nPutCardList.push_back(tmp_3);
                    clsHandCardData.value_nPutCardList.push_back(tmp_3);
                }
                clsHandCardData.uctPutCardType = BestCardGroup;
            }
            return;
        }
    }
    //3-2
    if (clsHandCardData.value_aHandCardList[16] == 1 && clsHandCardData.value_aHandCardList[17] == 0)
    {
        clsHandCardData.value_nPutCardList.push_back(16);
        clsHandCardData.uctPutCardType = get_GroupData(cgSINGLE, 16, 1);
        return;
    }
    if (clsHandCardData.value_aHandCardList[16] == 0 && clsHandCardData.value_aHandCardList[17] == 1)
    {
        clsHandCardData.value_nPutCardList.push_back(17);
        clsHandCardData.uctPutCardType = get_GroupData(cgSINGLE, 17, 1);
        return;  
    }
    //
    for (int i = 3; i < 16; i++)
    {
        if (clsHandCardData.value_aHandCardList[i] == 4)
        {
            clsHandCardData.value_nPutCardList.push_back(i);
            clsHandCardData.value_nPutCardList.push_back(i);
            clsHandCardData.value_nPutCardList.push_back(i);
            clsHandCardData.value_nPutCardList.push_back(i);
            
            clsHandCardData.uctPutCardType = get_GroupData(cgBOMB_CARD, i, 4);
            
            return;
        }  
    }  

    //
    clsHandCardData.uctPutCardType = get_GroupData(cgERROR, 0, 0);
    return;

}

/*
 dp
 get_PutCardList
 HandCardValue
 
 */

HandCardValue get_HandCardValue(HandCardData &clsHandCardData)
{
    
    //get_PutCardList
    clsHandCardData.ClearPutCardList();
    
    HandCardValue uctHandCardValue;
    //
    if (clsHandCardData.nHandCardCount == 0)
    {
        uctHandCardValue.SumValue = 0;
        uctHandCardValue.NeedRound = 0;
        return uctHandCardValue;
    }
    //
    CardGroupData uctCardGroupData = ins_SurCardsType(clsHandCardData.value_aHandCardList);
    //
    if (uctCardGroupData.cgType != cgERROR&&uctCardGroupData.cgType != cgFOUR_TAKE_ONE&&uctCardGroupData.cgType != cgFOUR_TAKE_TWO)
    {
        uctHandCardValue.SumValue = uctCardGroupData.nValue;
        uctHandCardValue.NeedRound = 1;
        return uctHandCardValue;
    }
    
    //
    
    /*clsHandCardData.value_nPutCardListclsHandCardData.uctPutCardType
     get_PutCardList*/
    get_PutCardList_2_unlimit(clsHandCardData);
    
    //clsHandCardData.value_nPutCardListclsHandCardData.uctPutCardType
    CardGroupData NowPutCardType = clsHandCardData.uctPutCardType;
    vector<int> NowPutCardList = clsHandCardData.value_nPutCardList;
    
    if (clsHandCardData.uctPutCardType.cgType == cgERROR)
    {
        cout << "PutCardType ERROR!" << endl;
    }
    
    
    
    //---
    for (vector<int>::iterator iter = NowPutCardList.begin();
         iter != NowPutCardList.end(); iter++)
    {
        clsHandCardData.value_aHandCardList[*iter]--;
    }
    clsHandCardData.nHandCardCount -= NowPutCardType.nCount;
    //---
    HandCardValue tmp_SurValue = get_HandCardValue(clsHandCardData);//
    
    //---
    for (vector<int>::iterator iter = NowPutCardList.begin();
         iter != NowPutCardList.end(); iter++)
    {
        clsHandCardData.value_aHandCardList[*iter]++;
    }
    clsHandCardData.nHandCardCount += NowPutCardType.nCount;
    //---
    
    uctHandCardValue.SumValue = NowPutCardType.nValue + tmp_SurValue.SumValue;
    uctHandCardValue.NeedRound = tmp_SurValue.NeedRound + 1;
     
    return uctHandCardValue;
}


/*
 2.0  
 */

void get_PutCardList_2(GameSituation &clsGameSituation, HandCardData &clsHandCardData)
{
    if (clsGameSituation.nCardDroit == clsHandCardData.nOwnIndex)
    {
        // printf("unlimit\n");
        get_PutCardList_2_unlimit(clsHandCardData);
    }
    else
    {
        // printf("limit\n");
        get_PutCardList_2_limit(clsGameSituation, clsHandCardData);
    }
    return;
}


//************************************
// Method:    get_all_actions
// FullName:  get_all_actions
// Access:    public 
// Returns:   std::vector<CardGroup>
// Qualifier: the action orders may not be the same as Python's
//************************************
// here cardData is a one hot representation of current hand data
void my_get_PutCardList_2(GameSituation &clsGameSituation, HandCardData &clsHandCardData) {
    if (clsGameSituation.nCardDroit == clsHandCardData.nOwnIndex)
    {
        // printf("unlimit\n");
        my_get_PutCardList_2_unlimit(clsHandCardData);
    }
    else
    {
        // printf("limit\n");
        my_get_PutCardList_2_limit(clsGameSituation, clsHandCardData);
    }
    return;
}
// find limit best card group
void my_get_PutCardList_2_limit(GameSituation &clsGameSituation, HandCardData &clsHandCardData) {
    // preparation
    clsHandCardData.ClearPutCardList();
    vector<int> cardData_vector = clsHandCardData.value_nHandCardList;
    int cardData[15];
    get_one_hot_representation(cardData, cardData_vector, false);
    // find best card group
    CardGroupType cg_type = clsGameSituation.uctNowCardGroup.cgType;
    int standard_len = clsGameSituation.uctNowCardGroup.nCount;
    int standard_max_card = clsGameSituation.uctNowCardGroup.nMaxCard;
    CardGroupNode best_node = find_best_group_limit(clsGameSituation, cardData);
    CardGroupType best_action_type = best_node.group_type;
    vector<int> best_group_data = best_node.group_data;
    vector<int> best_remain_cards = best_node.remain_cards;
    // put cards
    int action_max_card, action_len;
    get_card_group_max_and_len(best_group_data, (int) best_action_type, action_max_card, action_len);
    clsHandCardData.value_nPutCardList = best_group_data;
    clsHandCardData.uctPutCardType = clsGameSituation.uctNowCardGroup = get_GroupData(best_action_type, action_max_card, action_len);
    return;
}
// find unlimit best card group
void my_get_PutCardList_2_unlimit(HandCardData &clsHandCardData) {
    // preparation
    clsHandCardData.ClearPutCardList();
    vector<int> cardData_vector = clsHandCardData.value_nHandCardList;
    int cardData[15];
    get_one_hot_representation(cardData, cardData_vector, false);
    // find best card group
    CardGroupNode best_node = find_best_group_unlimit(cardData);
    CardGroupType best_action_type = best_node.group_type;
    vector<int> best_group_data = best_node.group_data;
    vector<int> best_remain_cards = best_node.remain_cards;
    // put cards
    int action_max_card, action_len;
    get_card_group_max_and_len(best_group_data, (int) best_action_type, action_max_card, action_len);
    clsHandCardData.value_nPutCardList = best_group_data;
    clsHandCardData.uctPutCardType = get_GroupData(best_action_type, action_max_card, action_len);
    return;
}

vector<CardGroup> get_all_actions_unlimit(int cardData[]) {
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
            for(int idx = 0; idx < 15; idx ++)
            {
                vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
                if(cardData[idx] >= 1 && idx != i)
                {
                    main_cards.push_back(Card(idx));
                    cardData[idx] -= 1;
                    for(int idx1 = 0; idx1 < 15; idx1 ++)
                    {
                        if(cardData[idx1] >= 1 && idx1 != i)
                        {
                            main_cards.push_back(Card(idx1));
                            actions.push_back(CardGroup(main_cards, Category::FOUR_TAKE_ONE, i));
                            main_cards.pop_back();
                        }
                    }
                    cardData[idx] += 1;
                }
            }
            for(int idx = 0; idx < 15; idx ++)
            {
                vector<Card> main_cards = { Card(i), Card(i), Card(i), Card(i) };
                if(cardData[idx] >= 2 && idx != i)
                {
                    main_cards.push_back(Card(idx));
                    main_cards.push_back(Card(idx));
                    cardData[idx] -= 2;
                    for(int idx1 = 0; idx1 < 15; idx1 ++)
                    {
                        if(cardData[idx1] >= 2 && idx1 != i)
                        {
                            main_cards.push_back(Card(idx1));
                            main_cards.push_back(Card(idx1));
                            actions.push_back(CardGroup(main_cards, Category::FOUR_TAKE_TWO, i));
                            main_cards.pop_back();
                            main_cards.pop_back();
                        }
                    }
                    cardData[idx] += 2;
                }
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
                tmp.push_back(Card(i + 3));
                if (!single && cardData[i] >= 2)
                {
                    tmp.push_back(Card(i + 3));
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

// change value_nHandCardList to a one hot representation
void get_one_hot_representation(int one_hot[], vector<int> hand_card_data, bool zero_start) {
    for(int idx = 0; idx < 15; idx ++) {
        int new_idx = idx;
        if(!zero_start) new_idx = idx + 3;
        for(int card:hand_card_data) {
            if(new_idx == card) one_hot[idx] += 1;
        }
    }
    return;
}

vector<vector<int>> cardGroupNode2matrix(vector<CardGroupNode> &card_group_nodes) {
    vector<vector<int>> card_group_matrix;
    for(CardGroupNode node:card_group_nodes) {
        card_group_matrix.push_back(node.group_data);
    }
    return card_group_matrix;
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

vector<int> one_card_group2vector(CardGroup card_group) {
    vector<int> vct;
    vector<Card> cg_cards = card_group._cards;
    for(Card cd:cg_cards) {
            vct.push_back((int) cd);
        }
    return vct;
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
    vector<CardGroup> all_actions = get_all_actions_unlimit(cardData);
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

CardGroupNode find_best_group_unlimit(int cardData[]){
    vector<CardGroup> card_groups = get_all_actions_unlimit(cardData); // 0-14
    vector<float> value_caches;
    for(CardGroup card_group:card_groups) {
        if(card_group._cards.size() != 0) {
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

CardGroupNode find_best_group_limit(GameSituation &clsGameSituation, int cardData[]) {
    vector<CardGroup> card_groups = get_all_actions_unlimit(cardData); // 0-14
    vector<float> value_caches;
    CardGroupType cg_type = clsGameSituation.uctNowCardGroup.cgType;
    for(CardGroup card_group:card_groups) {
        if(is_legal(clsGameSituation, card_group) && card_group._cards.size() != 0) {
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

void get_card_group_max_and_len(vector<int> &action, int standard_type, int &action_max_card, int &action_len) {
    vector<int> cache;
    int a;
    switch(standard_type) {
        case 1:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 2:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 3:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 4:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 5:
        {
            action_len = action.size();
            for(int a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 3) {
                    action_max_card = a;
                    break;
                }
            }
            assert(action_max_card >= 0);
        }
        case 6:
        {
            action_len = action.size();
            for(int a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 3) {
                    action_max_card = a;
                    break;
                }
            }
            assert(action_max_card >= 0);
        }
        case 7:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 8:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 9:
        {
            action_len = action.size();
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 10:
        {
            action_len = action.size();
            vector<int> cache;
            for(int a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 3 && find(cache.begin(), cache.end(), a) == cache.end()) {
                    cache.push_back(a);
                }
                if(cache.size() == 2) break;
            }
            action_max_card = *max_element(cache.begin(), cache.end());
        }

        case 11:
        {
            action_len = action.size();
            for(a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 3 && find(cache.begin(), cache.end(), a) == cache.end()) {
                    cache.push_back(a);
                }
                if(cache.size() == 2) break;
            }
            action_max_card = *max_element(cache.begin(), cache.end());
        }
        case 12:
        {
            action_len = 2;
            action_max_card = *max_element(action.begin(), action.end());
        }
        case 13:
        {
            action_len = 6;
            for(a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 4) {
                    action_max_card = a;
                    break;
                }
            }
            assert(action_max_card >= 0);
        }
        case 14:
        {
            action_len = 10;
            for(a:action) {
                int times = count(action.begin(), action.end(), a);
                if(times == 4) {
                    action_max_card = a;
                    break;
                }
            }
            assert(action_max_card >= 0);
        }
    }
}

bool is_legal(GameSituation &clsGameSituation, CardGroup &candidate_action) {
    int standard_type = (int) clsGameSituation.uctNowCardGroup.cgType;
    int n_count = clsGameSituation.uctNowCardGroup.nCount;
    int max_card = clsGameSituation.uctNowCardGroup.nMaxCard;
    int action_len = -1;
    int action_max_card = -1;
    vector<int> action = one_card_group2vector(candidate_action);
    int action_type = (int) candidate_action._category;
    if(!(standard_type == action_type)) return false;
    else {
        get_card_group_max_and_len(action, standard_type, action_max_card, action_len);
        if(action_len == n_count && action_max_card > max_card) return true;
        else return false;
    }
}


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


