#include <iostream>
#include<string>
using namespace std;
#include"speechManger.h"
int main()
{
	//�������������
	SpeechManger sm;

	//����12��ѡ�ֵĴ���
	//for (map<int, Speaker>::iterator it = sm.m_Speaker.begin();it != sm.m_Speaker.end();it++)
	//{
	//	cout << "ѡ�ֱ�ţ� " << it->first << " ѡ������: " << it->second.m_Name << " ����: " << it->second.m_Score[0] << endl;
	//}

	int choice = 0;
	while (true)
	{
		sm.show_Menu();
		cout << "����������ѡ��: ";
		cin >> choice;

		switch (choice)
		{
		case 1://��ʼ����
			sm.startSpeech();

			system("cls");//��������
			break;
		case 2://�鿴���������¼

			break;
		case 3://��ձ�����¼

			break;
		case 0://�˳�ϵͳ
			sm.exitSystem();
			break;
		default:
			system("cls");//��������
			break;
		}
	}
	system("pause");
	return 0;
}