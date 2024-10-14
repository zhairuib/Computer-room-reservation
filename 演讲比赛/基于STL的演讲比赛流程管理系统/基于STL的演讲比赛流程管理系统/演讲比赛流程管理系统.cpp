#include <iostream>
#include<string>
using namespace std;
#include"speechManger.h"
int main()
{
	//创建管理类对象
	SpeechManger sm;

	//测试12名选手的创建
	//for (map<int, Speaker>::iterator it = sm.m_Speaker.begin();it != sm.m_Speaker.end();it++)
	//{
	//	cout << "选手编号： " << it->first << " 选手姓名: " << it->second.m_Name << " 分数: " << it->second.m_Score[0] << endl;
	//}

	int choice = 0;
	while (true)
	{
		sm.show_Menu();
		cout << "请输入您的选择: ";
		cin >> choice;

		switch (choice)
		{
		case 1://开始比赛
			sm.startSpeech();

			system("cls");//清屏操作
			break;
		case 2://查看往届比赛记录

			break;
		case 3://清空比赛记录

			break;
		case 0://退出系统
			sm.exitSystem();
			break;
		default:
			system("cls");//清屏操作
			break;
		}
	}
	system("pause");
	return 0;
}