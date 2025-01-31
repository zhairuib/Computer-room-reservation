#include"manager.h"

//默认构造
Manager::Manager()
{

}

//有参构造
Manager::Manager(string name, string pwd)
{
	this->m_UserName = name;
	this->m_UserPassword = pwd;

	//初识化容器 获取到所有文件中 学生、老师、信息
	this->initVector();

	//初始化机房信息
	ifstream ifs;
	ifs.open(COMPUTER_FILE, ios::in);

	computerRoom com;
	while (ifs >> com.m_ComId && ifs >> com.m_ManNum)
	{
		vCom.push_back(com);
	}
	ifs.close();



}

//菜单
void Manager::operMenu()
{
	cout << "欢迎管理员:" << this->m_UserName << "登录!" << endl;
	cout << "——————————" << endl;
	cout << "——1.添加账号——" << endl;
	cout << "——2.查看账号——" << endl;
	cout << "——3.查看机房——" << endl;
	cout << "——4.清空预约——" << endl;
	cout << "——0.注销登录——" << endl;
	cout << "——————————" << endl;
	cout << "——————————" << endl;
	cout << "当前学生数量:" << vStu.size() << endl;
	cout << "当前老师数量:" << vTea.size() << endl;
	cout << "机房数量为:" << vCom.size() << endl;
	cout << "——————————" << endl;
	cout << "请选择您的操作:" << endl;
}

//添加账号
void Manager::addPerson()
{
	cout << "请输入添加账号的类型" << endl;
	cout << "1.添加学生" << endl;
	cout << "2.添加老师" << endl;

	string fileName;//操作文件名
	string tip;//提示id号
	ofstream ofs;//文件操作对象

	string errorTip;

	int select = 0;
	cin >> select;
	if (select == 1)
	{
		fileName = STUDENT_FILE;
		tip = "请输入学号";
		errorTip = "学号重复,重新输入!";
	}
	else
	{
		fileName = TEACHER_FILE;
		tip = "请输入职工编号";
		errorTip = "职工号重复,重新输入!";
	}

	ofs.open(fileName, ios::out | ios::app);
	int id;
	string name;
	string pwd;


	cout << tip << endl;

	while (1)
	{
		cin >> id;
		bool ret = checkRepeat(id, select);
		if (ret)//有重复
		{
			cout << errorTip << endl;
		}
		else
		{
			break;
		}
	}

	cout << "请输入姓名" << endl;
	cin >> name;

	cout << "请输入密码" << endl;
	cin >> pwd;

	//向文件添加数据

	ofs << id << " " << name << " " << pwd << " " << endl;
	cout << "添加成功" << endl;



	system("pause");
	system("cls");

	ofs.close();

	this->initVector();
}

void printStudent(Student& s)
{
	cout << "学号:" << s.m_Id << " 姓名:" << s.m_UserName << " 密码:" << s.m_UserPassword << endl;
}

void printTeacher(Teacher& s)
{
	cout << "职工号:" << s.m_EmpId << " 姓名:" << s.m_UserName << " 密码:" << s.m_UserPassword << endl;
}


//查看账号
void Manager::showPerson()
{
	cout << "请选择查看内容" << endl;
	cout << "1.查看所有学生" << endl;
	cout << "2.查看所有老师" << endl;

	int select = 0;
	cin >> select;

	if (select == 1)
	{
		cout << "所有学生信息如下:" << endl;
		for_each(vStu.begin(), vStu.end(), printStudent);
	}
	else
	{
		cout << "所老师生信息如下:" << endl;
		for_each(vTea.begin(), vTea.end(), printTeacher);
	}

	system("pause");
	system("cls");
}

//查看机房信息
void Manager::showComputer()
{
	cout << "机房信息如下" << endl;
	for (vector<computerRoom>::iterator it = vCom.begin(); it != vCom.end(); it++)
	{
		cout << "机房编号:" << it->m_ComId << " 机房最大容量:" << it->m_ManNum << endl;
	}

	system("pause");
	system("cls");
}

//清空预约记录
void Manager::cleamFile()
{
	ofstream ofs(ORDER_FILE, ios::trunc);
	ofs.close();

	cout << "清空成功" << endl;
	system("pause");
	system("cls");
}

//初始化容器
void Manager::initVector()
{
	//读取信息 学生 老师 
	ifstream ifs;
	ifs.open(STUDENT_FILE, ios::in);
	if (!ifs.is_open())
	{
		cout << "文件读取失败" << endl;
		return;
	}

	vStu.clear();
	vTea.clear();

	Student s;
	while (ifs >> s.m_Id && ifs >> s.m_UserName && ifs >> s.m_UserPassword)
	{
		vStu.push_back(s);
	}

	ifs.close();

	//读取信息 老师
	ifs.open(TEACHER_FILE, ios::in);
	Teacher t;
	while (ifs >> t.m_EmpId && ifs >> t.m_UserName && ifs >> t.m_UserPassword)
	{
		vTea.push_back(t);
	}
	ifs.close();
}

//检测重复 
bool Manager::checkRepeat(int id, int type)
{
	if (type == 1)
	{
		for (vector<Student>::iterator it = vStu.begin(); it != vStu.end(); it++)
		{
			if (id == it->m_Id)
			{
				return true;
			}
		}
	}
	else
	{
		for (vector<Teacher>::iterator it = vTea.begin(); it != vTea.end(); it++)
		{
			if (id == it->m_EmpId)
			{
				return true;
			}
		}
	}

	return false;
}
