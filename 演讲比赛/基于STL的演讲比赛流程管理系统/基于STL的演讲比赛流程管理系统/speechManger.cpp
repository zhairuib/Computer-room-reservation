#include "speechManger.h"
#include<algorithm>


//构造函数
SpeechManger::SpeechManger() 
{
	//初始化容器和属性
	this->initSpeech();
	//创建选手
	this->createSpeaker();
}

//菜单函数
void SpeechManger::show_Menu()
{
	cout << "******************************" << endl;
	cout << "*******欢迎参加演讲比赛*******" << endl;
	cout << "*********1.开始比赛***********" << endl;
	cout << "*********2.查看记录***********" << endl;
	cout << "*********3.清空记录***********" << endl;
	cout << "*********0.退出程序***********" << endl;
	cout << "******************************" << endl;
	cout << endl;
}

//退出系统
void SpeechManger::exitSystem()
{
	cout << "欢迎下次使用" << endl;
	system("pause");
	exit(0);
}

//初始化容器和属性
void SpeechManger::initSpeech()
{
	//容器都置空
	this->v1.clear();
	this->v2.clear();
	this->vVictory.clear();
	this->m_Speaker.clear();

	//初始化比赛轮数
	this->m_Index = 1;

}


//创建12名选手
void SpeechManger::createSpeaker()
{
	string nameSeed = "ABCDEFGHIJKL";
	for (int i = 0;i < nameSeed.size();i++)
	{
		string name = "选手";
		name += nameSeed[i];

		Speaker sp;
		sp.m_Name = name;

		for (int j = 0;j < 2;j++)
		{
			sp.m_Score[j] = 0;
		}

		//创建选手编号，并放到v1容器中
		this->v1.push_back(i + 1001);

		//选手编号以及对应选手 放入到map容器中
		this->m_Speaker.insert(make_pair(i + 1001, sp));
	}
}

//开始比赛 比赛整个流程控制函数
void SpeechManger::startSpeech()
{
	//第一轮开始比赛

	//1.抽签
	this->speechDraw();
	//2.比赛
	this->speechContest();
	//3.显示晋级结果

	//第二轮开始比赛

	//1.抽签

	//2.比赛

	//3.显示最终结果

	//4.保存分数到文件中

}

//抽签
void SpeechManger::speechDraw()
{
	cout << "第 << " << this->m_Index << " >> 轮比赛选手正在抽签" << endl;
	cout << "-----------------------" << endl;
	cout << "抽签后的演讲顺序如下： " << endl;

	if (this->m_Index == 1)
	{
		//第一轮比赛
		random_shuffle(v1.begin(), v1.end());
		for (vector<int>::iterator it = v1.begin();it != v1.end();it++)
		{
			cout << *it << " ";
		}
		cout << endl;
	}
	else
	{
		//第二轮比赛
		random_shuffle(v2.begin(), v2.end());
		for (vector<int>::iterator it = v2.begin();it != v2.end();it++)
		{
			cout << *it << " ";
		}
		cout << endl;
	}
	cout << "-----------------------" << endl;
	system("pause");
}

//比赛
void SpeechManger::speechContest()
{
	cout << "————第 << " << this->m_Index << " >> 轮比赛正式开始————" << endl;
	cout << endl;
	//准备临时容器 存放小组成绩 
	multimap<double, int, greater<double>>groupScore;

	int num = 0;//6人一组

	vector<int>v_Src;//比赛选手容器
	if (this->m_Index == 1)
	{
		v_Src = v1;
	}
	else
	{
		v_Src = v2;
	}
	for (vector<int>::iterator it = v_Src.begin();it != v_Src.end();it++)
	{
		num++;
		//评委打分
		deque <double>d;
		for (int i = 0;i < 10;i++)
		{
			double score = (rand() % 401 + 600) / 10.f;//600-1000
			//cout << score << " ";
			d.push_back(score);
		}
		//cout << endl;
		sort(d.begin(), d.end(), greater<double>());
		d.pop_back();
		d.pop_front();

		double sum = accumulate(d.begin(), d.end(), 0.0f);//总分
		double avg = sum / (double)d.size();//平均分

		//打印平均分
		//cout << "编号: " << *it << "     姓名： " << this->m_Speaker[*it].m_Name << "      获取平均分: " << avg << endl;

		//将平均分放入到map容器里
		this->m_Speaker[*it].m_Score[this->m_Index - 1] = avg;

		//将打分数据 放入到临时小组容器中
		groupScore.insert(make_pair(avg, *it));//(得分，选手编号)

		//每6人取出前三名
		if (num % 6 == 0)
		{
			cout << "第 " << num / 6 << " 小组比赛名次： " << endl;

			//取走前三名
			int count = 0;
			for (multimap<double, int, greater<double>>::iterator 
				it = groupScore.begin();
				it != groupScore.end(); 
				it++)
			{
				cout << "编号： " << it->second << " 姓名： " << this->m_Speaker[it->second].m_Name
					<< " 成绩： " << m_Speaker[it->second].m_Score[this->m_Index - 1] << endl;
			}

			for (multimap<double, int, greater<double>>::iterator
				it = groupScore.begin();
				it != groupScore.end() && count < 3;
				it++, count++)
			{
				if (this->m_Index == 1)
				{
					v2.push_back((*it).second);
				}
				else
				{
					vVictory.push_back((*it).second);
				}
			}

			groupScore.clear();//小组清空
			cout << endl;
		}

	}
	cout << endl;
	cout << "--------- 第 " << this->m_Index << " 轮比赛完毕! --------" << endl;
	system("pause");
}

//析构函数
SpeechManger::~SpeechManger()
{

}