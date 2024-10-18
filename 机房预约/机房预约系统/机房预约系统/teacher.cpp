#include"teacher.h"


Teacher::Teacher()
{

}

//�вι���
Teacher::Teacher(int empId, string name, string pwd)
{
	//��ʼ������
	this->m_EmpId = empId;
	this->m_UserName = name;
	this->m_UserPassword = pwd;

}


//�˵�
void Teacher::operMenu()
{
	cout << "��ӭ��ʦ: " << this->m_UserName << "��¼! " << endl;
	cout << "��������������������" << endl;
	cout << "����1.�鿴����ԤԼ����" << endl;
	cout << "����2.���ԤԼ��������" << endl;
	cout << "����0.ע����¼������" << endl;
	cout << "��������������������" << endl;
	cout << "��ѡ����Ĳ���" << endl;
}

//�鿴����ԤԼ
void Teacher::showAllOrder()
{
	OrderFile of;
	if (of.m_Size == 0)
	{
		cout << "��ԤԼ��¼" << endl;
		system("pause");
		system("cls");
		return;
	}
	for (int i = 0; i < of.m_Size; i++)
	{
		cout << i + 1 << "�� ";
		cout << "ԤԼ����: ��" << of.m_orderData[i]["date"];
		cout << " ʱ���:  " << (of.m_orderData[i]["interval"] == "1" ? "����" : "����");
		cout << " ѧ��:  " << of.m_orderData[i]["stuId"];
		cout << " ����:  " << of.m_orderData[i]["stuName"];
		cout << " �������:  " << of.m_orderData[i]["roomId"];
		string status = " ״̬: ";

		if (of.m_orderData[i]["status"] == "1")
		{
			status += "�����";
		}
		else if (of.m_orderData[i]["status"] == "2")
		{
			status += "ԤԼ�ɹ�";
		}
		else if (of.m_orderData[i]["status"] == "-1")
		{
			status += "ԤԼʧ�ܣ����δͨ��";
		}
		else
		{
			status += "ԤԼ��ȡ��";
		}
		cout << status << endl;
	}
	system("pause");
	system("cls");
}

//���ԤԼ
void Teacher::validOrder()
{
	OrderFile of;
	if (of.m_Size == 0)
	{
		cout << "��ԤԼ��¼" << endl;
		system("pause");
		system("cls");
		return;
	}

	cout << "����˵�ԤԼ��¼����:" << endl;

	vector<int>v;

	int index = 0;
	for (int i = 0; i < of.m_Size; i++)
	{
		if (of.m_orderData[i]["status"] == "1")
		{
			v.push_back(i);
			cout << ++index << "�� ";
			cout << "ԤԼ����: ��" << of.m_orderData[i]["date"];
			cout << " ʱ���: " << (of.m_orderData[i]["interval"] == "1" ? "����" : "����");
			cout << "ѧ�����: " << of.m_orderData[i]["stuId"];
			cout << "ѧ������: " << of.m_orderData[i]["stuName"];
			cout << "�������: " << of.m_orderData[i]["roomId"];
			cout << "״̬: ����� " << endl;
		}
	}
	cout << "������Ҫ��˵�ԤԼ��¼,0������" << endl;
	int select = 0;//�����û�ѡ���ԤԼ��¼
	int ret = 0;//����ԤԼ�����¼
	while (1)
	{
		cin >> select;
		if (select >= 0 && select <= v.size())
		{
			if (select == 0)
			{
				break;
			}
			else
			{
				cout << "��������˽��" << endl;
				cout << "1.ͨ��" << endl;
				cout << "2.��ͨ��" << endl;
				cin >> ret;
				if (ret == 1)
				{
					//ͨ��
					of.m_orderData[v[select - 1]]["status"] = "2";//-1����Ϊvector�����±��Ǵ�0��ʼ��
				}
				else
				{
					//��ͨ��
					of.m_orderData[v[select - 1]]["status"] = "-1";
				}
				//����ԤԼ��¼
				of.updateOrder();
				cout << "������" << endl;
				break;
			}
		}
		cout << "������������������" << endl;
	}
	system("pause");
	system("cls");
}

