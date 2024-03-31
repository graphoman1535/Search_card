#include <opencv2/opencv.hpp>
#include <iostream>



int  findIntersectionPoint(cv::Vec4f line1, cv::Vec4f line2, cv::Point* crossPoint);
void drawline(cv::Mat img, cv::Vec2f line);
void drawline(cv::Mat img, cv::Vec4f line);
double calculateAngleBetweenLines(cv::Vec4f line1, cv::Vec4f line2);
int isIntersection(cv::Vec4i line1, cv::Vec4i line2);
double approximateDistanceBetweenAlmostParallelLines(cv::Vec4f line1, cv::Vec4f line2);
double distanceBetweenParallelSegments(cv::Vec4f segment1, cv::Vec4f segment2);
double pointToLineDistance(double x, double y, double A, double B, double C);
bool isSegmentBetweenPoints(cv::Vec4i segment, cv::Point pt1, cv::Point pt2);
int orientation(cv::Point pt, cv::Vec4i line);
int distanceSquared(const cv::Point& p1, const cv::Point& p2);
bool isCard(const std::vector<cv::Point>& points);


int main(int argc, char** argv) {
	// Загрузка изображения
	cv::Mat img = cv::imread("../im3.jpg");
	if (img.empty()) {
		std::cout << "Image Not Found!!!" << std::endl;
		return -1;
	}

	double scale_factor = 0.2;
	int new_height = cvRound(img.rows * scale_factor);
	int new_width = cvRound(img.cols * scale_factor);
	cv::resize(img, img, cv::Size(new_width, new_height));

	// Уменьшение шума
	GaussianBlur(img, img, cv::Size(7, 7), 1.5);

	// Конвертация в градации серого
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// Применение детектора границ Canny
	cv::Mat cany_res;
	Canny(gray, cany_res, 50, 150);
	//cv::imwrite("cany.jpg", cany_res);

// Вероятностный Хофф
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(cany_res, lines, 2, CV_PI / 180, 75, 30, 15);

	// -Ищем пересекающися отрезки.	-----
	std::set<int> indexes;
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i; j < lines.size(); j++) {
			int intersection = isIntersection(lines[i], lines[j]);
			if (intersection == 1)
			{
				indexes.insert(i);
				indexes.insert(j);
			}
		}
	}
	// Удаляем пересекающиеся отрезки
	for (auto it = indexes.rbegin(); it != indexes.rend(); ++it) {
		if (*it < lines.size()) {
			lines.erase(lines.begin() + *it);
		}
	}
	// удалим близкие отрезки. 
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i; j < lines.size(); j++) {
			cv::Vec2f line1, line2;
			double angleBetweenLines = calculateAngleBetweenLines(lines[i], lines[j]);

			cv::Point crossPoint;
			int status_line = findIntersectionPoint(lines[i], lines[j], &crossPoint);

			if (status_line == -1)// прямые паралельны
			{
				double distance = distanceBetweenParallelSegments(lines[i], lines[j]);
				if (distance < 20 && distance>0)
					lines.erase(lines.begin() + j);
				continue;
			}
			// если прямые близки к паралельным и находятся рядом.
			if (angleBetweenLines > 0 && angleBetweenLines < 10 * (CV_PI / 180)) {
				double aprox_distance = approximateDistanceBetweenAlmostParallelLines(lines[i], lines[j]);
				if (aprox_distance > 0 && aprox_distance < 20)
					lines.erase(lines.begin() + j);
				continue;

			}

			// Проверка что точка пересечения в области картинки.
			bool isInside = (crossPoint.x >= 0) && (crossPoint.x < img.cols) && (crossPoint.y >= 0) && (crossPoint.y < img.rows);

			if (!isInside) {// точка вне изображения
				continue;
			}

			if (angleBetweenLines > 0 && angleBetweenLines < 10 * (CV_PI / 180))
			{
				lines.erase(lines.begin() + j);
			}

		}
	}

	int counter1 = 0, counter2 = 0;

	std::vector<cv::Vec4i> lines1_for_point;
	std::vector<cv::Vec4i> lines2_for_point;
	std::vector<cv::Point> future_points;
	// ищем точки персечения прямых которые пересекаются под прямым углом 
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i + 1; j < lines.size(); j++) {
			cv::Point crossPoint;
			int property = findIntersectionPoint(lines[i], lines[j], &crossPoint);
			if (property == -1)// прямые паралельны
			{
				continue;
			}
			// Проверка что точка пересечения в области картинки.
			bool isInside = (crossPoint.x >= 0) && (crossPoint.x < img.cols) && (crossPoint.y >= 0) && (crossPoint.y < img.rows);

			if (!isInside) {// точка вне изображения
				continue;
			}

			// нужно чтобы прямые пересекались примерно под прямым углом
			double angleBetweenLines = calculateAngleBetweenLines(lines[i], lines[j]);
			if (angleBetweenLines > CV_PI / 2 + 3 * (CV_PI / 180) || angleBetweenLines < CV_PI / 2 - 3 * (CV_PI / 180))
			{
				continue;
			}

			uchar weight_sum = 0;
			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					int newX = crossPoint.x + k;
					int newY = crossPoint.y + l;
					weight_sum += cany_res.at<uchar>(newY, newX);
				}
			}

			if (weight_sum == 0) {//Точки нет на картинке
				for (int k = -1; k < 2; k++) {
					for (int l = -1; l < 2; l++) {
						int newX = crossPoint.x + k;
						int newY = crossPoint.y + l;
						cany_res.at<uchar>(newY, newX) = 255;
					}
				}
				future_points.push_back(crossPoint);
				lines1_for_point.push_back(lines[i]);
				lines2_for_point.push_back(lines[j]);
			}
		}
	}
	
	// посмотрим различные комбинации точек. Могут ли они образовывать прямоугольник 
	std::set<int> index_points;

	for (size_t i = 0; i < future_points.size(); ++i) {
		for (size_t j = i + 1; j < future_points.size(); ++j) {

			std::vector<cv::Vec4i> generatingLines = {
				lines1_for_point[i], lines2_for_point[i], lines1_for_point[j],lines2_for_point[j] };

			// есть ли среди образующих две точки общие прямых.
			bool foundEqual = false;
			for (size_t k = 0; k < generatingLines.size() && !foundEqual; ++k) {
				for (size_t l = k + 1; l < generatingLines.size(); ++l) {
					if (generatingLines[k] == generatingLines[l]) {
						foundEqual = true;
						std::vector<int> number_line = { 0,1,2,3 };
						number_line.erase(number_line.begin() + std::max(k,l));
					    number_line.erase(number_line.begin() + std::min(k,l));
						
						// проверим лежит ли образующий отрезок между точками. (У карточки обязательно лежит )
						if (isSegmentBetweenPoints(generatingLines[k], future_points[i], future_points[j]))
						{
							int orient1 = orientation(future_points[i], generatingLines[number_line[0]]);
							int orient2 = orientation(future_points[j], generatingLines[number_line[1]]);
				
							if (orient1 == orient2) // не совпадающие прямые должны лежать по одну сторону от точек
							{
								//double
								//	dist_betwen_line = approximateDistanceBetweenAlmostParallelLines(generatingLines[number_line[0]], generatingLines[number_line[1]]),
								//	real_dist =sqrt(distanceSquared(future_points[i], future_points[j]));
								//std::cout <<"dist_betwen_line =" <<dist_betwen_line << std::endl;
								//std::cout << "dist_betwen_points =" << real_dist << std::endl;
								//
								//if (fabs(dist_betwen_line -real_dist) < 2) { // расстояние между вершинами должно быть равно расстоянию между паралельными прямыми
								//	index_points.insert(i);
								//	index_points.insert(j);
								//	break;
								//}
								index_points.insert(i);
								index_points.insert(j);
							}
							break;
						}
						break;
					}
				}
			}
		}
	}
	std::cout << index_points.size() << std::endl;

	for (std::set<int>::iterator it = index_points.begin(); it != index_points.end(); ++it) {
		cv::circle(img, future_points[*it], 3, cv::Scalar(0, 255, 0), 1);
		drawline(img, lines1_for_point[*it]);
		drawline(img, lines2_for_point[*it]);
	}
	cv::imshow("please", img);
	cv::waitKey(0);
	std::vector<int> index_point_vec;//(index_points.begin(), index_points.end());

	for (std::set<int>::iterator it = index_points.begin(); it != index_points.end(); ++it) {
		index_point_vec.push_back(*it);
	}


	if (index_point_vec.size() != 4) {
		//переберем всевозможные точки. Можно ли получить из них прямоугольник.
		for (size_t i = 0; i < index_point_vec.size(); ++i) {
			for (size_t j = i + 1; j < index_point_vec.size(); ++j) {
				for (size_t k = j + 1; k < index_point_vec.size(); ++k) {
					for (size_t l = k + 1; l < index_point_vec.size(); ++l) {

						std::vector<cv::Point> vertex;
						vertex.push_back(future_points[index_point_vec[i]]);
						vertex.push_back(future_points[index_point_vec[j]]);
						vertex.push_back(future_points[index_point_vec[k]]);
						vertex.push_back(future_points[index_point_vec[l]]);
						;
						if (isCard(vertex)) {

							circle(img,future_points[index_point_vec[i]], 3, cv::Scalar(0, 255, 0), 1);
							circle(img,future_points[index_point_vec[j]], 3, cv::Scalar(0, 255, 0), 1);
							circle(img,future_points[index_point_vec[k]], 3, cv::Scalar(0, 255, 0), 1);
							circle(img,future_points[index_point_vec[l]], 3, cv::Scalar(0, 255, 0), 1);


						}




					}
				}
			}
		}
	}
	//cv::imshow("please", img);
	//cv::waitKey(0);


	return 0;
}

//-------------------------------------------------------------------------
// Вспомогательные функции.
int distanceSquared(const cv::Point& p1, const cv::Point& p2) {
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Функция проверки, могут ли 4 точки образовать прямоугольник

bool isCard(const std::vector<cv::Point>& points) {
	if (points.size() != 4) return false;

	std::vector<int> dists;
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			dists.push_back(distanceSquared(points[i], points[j]));
		}
	}

	std::sort(dists.begin(), dists.end());

	bool isRectangle = (dists[0] - dists[1]<1)  && (dists[2] - dists[3] < 1) && (dists[4] - dists[5] < 1) && (dists[0] - dists[4] < 1);
	if (!isRectangle) return false;

	// Вычисляем соотношение сторон прямоугольника
	double sideRatio = std::sqrt(static_cast<double>(dists[0]) / dists[2]);
	std::cout << "sideRatio = " << sideRatio << std::endl;
	if ( (sideRatio > 0.62 && sideRatio < 0.635))
		return true;
	return false;
}






int orientation(cv::Point pt, cv::Vec4i line) {
	cv::Point
		segLeft(std::min(line[0], line[2]), std::min(line[1], line[3])),
		segRight(std::max(line[0], line[2]), std::max(line[1], line[3]));

	if (segRight.x - segLeft.x <= 2) // прямая вертикальная
	{
		if ( segRight.y <= pt.y) // прямая левее точки.
			return -1;

		else if ( segLeft.y >= pt.y) // прямая правее точки
			return 0;
		else
			return 1;
	}
	if (segRight.y - segLeft.y <= 2) // прямая горизонтальная
	{
		if (segRight.x <= pt.x) // прямая левее точки.
			return -1;

		else if (segLeft.x >= pt.x) // прямая правее точки
			return 0;
		else
			return 1;
	}



	if (segRight.x <= pt.x && segRight.y <= pt.y) // прямая левее точки.
		return -1;

	else if (segLeft.x >= pt.x && segLeft.y >= pt.y) // прямая правее точки
		return 0;
	else
		return 1;

}

bool isSegmentBetweenPoints(cv::Vec4i segment, cv::Point pt1, cv::Point pt2) {
	// Конечные точки сегмента
	cv::Point segStart(segment[0], segment[1]), segEnd(segment[2], segment[3]);

	// Определяем минимальные и максимальные значения координат среди заданных точек
	int minX = std::min(pt1.x, pt2.x);
	int maxX = std::max(pt1.x, pt2.x);
	int minY = std::min(pt1.y, pt2.y);
	int maxY = std::max(pt1.y, pt2.y);

	// Проверка для вертикальной линии (координаты X сегмента равны)
	if (segStart.x == segEnd.x && (segStart.x >= minX && segStart.x <= maxX)) {
		return (segStart.y >= minY && segStart.y <= maxY && segEnd.y >= minY && segEnd.y <= maxY);
	}
	// Проверка для горизонтальной линии (координаты Y сегмента равны)
	else if (segStart.y == segEnd.y && (segStart.y >= minY && segStart.y <= maxY)) {
		return (segStart.x >= minX && segStart.x <= maxX && segEnd.x >= minX && segEnd.x <= maxX);
	}
	// Проверка для невертикальных и негоризонтальных линий
	else {
		return (segStart.x >= minX && segStart.x <= maxX && segStart.y >= minY && segStart.y <= maxY) &&
			(segEnd.x >= minX && segEnd.x <= maxX && segEnd.y >= minY && segEnd.y <= maxY);
	}
}


double pointToLineDistance(double x, double y, double A, double B, double C) {
	return std::abs(A * x + B * y + C) / std::sqrt(A * A + B * B);
}

double distanceBetweenParallelSegments(cv::Vec4f segment1, cv::Vec4f segment2) {
	double x1 = segment1[0], y1 = segment1[1], x2 = segment1[2], y2 = segment1[3];
	double x3 = segment2[0], y3 = segment2[1], x4 = segment2[2], y4 = segment2[3];

	// Вычисляем коэффициенты A, B, и C уравнения прямой, проходящей через две точки первого отрезка
	double A1 = y2 - y1, B1 = x1 - x2, C1 = -(A1 * x1 + B1 * y1);

	// Вычисляем коэффициенты A, B, и C уравнения прямой, проходящей через две точки второго отрезка
	double A2 = y4 - y3, B2 = x3 - x4, C2 = -(A2 * x3 + B2 * y3);

	// Вычисляем расстояния от конечных точек одного отрезка до прямой, заданной другим отрезком
	double distances[] = {
		pointToLineDistance(x1, y1, A2, B2, C2),
		pointToLineDistance(x2, y2, A2, B2, C2),
		pointToLineDistance(x3, y3, A1, B1, C1),
		pointToLineDistance(x4, y4, A1, B1, C1)
	};

	// Находим минимальное из вычисленных расстояний
	double minDistance = *std::min_element(distances, distances + 4);

	return minDistance;
}

double approximateDistanceBetweenAlmostParallelLines(cv::Vec4f line1, cv::Vec4f line2) {
	// Координаты первого отрезка
	double x1 = line1[0], y1 = line1[1];

	// Вычисляем коэффициенты A, B и C для уравнения прямой, заданной вторым отрезком в форме Ax + By + C = 0
	double x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
	double A2 = y4 - y3;
	double B2 = x3 - x4;
	double C2 = -A2 * x3 - B2 * y3;

	// Вычисляем расстояние от точки (x1, y1) первого отрезка до прямой, заданной вторым отрезком
	double distance = std::abs(A2 * x1 + B2 * y1 + C2) / std::sqrt(A2 * A2 + B2 * B2);

	return distance;
}

int isIntersection(cv::Vec4i line1, cv::Vec4i line2) {
	// Проверяем на вертикальность одной из линий
	bool line1Vertical = line1[2] == line1[0];
	bool line2Vertical = line2[2] == line2[0];

	double x_intersection, y_intersection;

	if (line1Vertical && line2Vertical) {
		// Обе линии вертикальны
		return 0; // Вертикальные линии не могут пересекаться, если они не совпадают
	}
	else if (line1Vertical || line2Vertical) {
		// Одна из линий вертикальна
		if (line1Vertical) {
			x_intersection = line1[0]; // X-координата вертикальной линии line1
			// Для вычисления y_intersection используем уравнение line2
			double k2 = (double)(line2[3] - line2[1]) / (line2[2] - line2[0]);
			double b2 = line2[1] - k2 * line2[0];
			y_intersection = k2 * x_intersection + b2;
		}
		else { // line2 вертикальна
			x_intersection = line2[0]; // X-координата вертикальной линии line2
			// Для вычисления y_intersection используем уравнение line1
			double k1 = (double)(line1[3] - line1[1]) / (line1[2] - line1[0]);
			double b1 = line1[1] - k1 * line1[0];
			y_intersection = k1 * x_intersection + b1;
		}
	}
	else {
		// Ни одна из линий не вертикальна
		double k1 = (double)(line1[3] - line1[1]) / (line1[2] - line1[0]);
		double b1 = line1[1] - k1 * line1[0];
		double k2 = (double)(line2[3] - line2[1]) / (line2[2] - line2[0]);
		double b2 = line2[1] - k2 * line2[0];

		if (std::fabs(k1 - k2) < 1e-8) { // Проверка на параллельность
			return 0;
		}

		x_intersection = (b2 - b1) / (k1 - k2);
		y_intersection = k1 * x_intersection + b1;
	}

	// Определение границ отрезков для проверки пересечения
	double x_min = std::max(std::min(line1[0], line1[2]), std::min(line2[0], line2[2])),
		x_max = std::min(std::max(line1[0], line1[2]), std::max(line2[0], line2[2])),
		y_min = std::max(std::min(line1[1], line1[3]), std::min(line2[1], line2[3])),
		y_max = std::min(std::max(line1[1], line1[3]), std::max(line2[1], line2[3]));

	if (x_intersection >= x_min && x_intersection <= x_max &&
		y_intersection >= y_min && y_intersection <= y_max) {
		return 1;
	}
	return 0;
}

double calculateAngleBetweenLines(cv::Vec4f line1, cv::Vec4f line2) {
	// Вычисляем направляющие вектора для каждой линии
	double dx1 = line1[2] - line1[0];
	double dy1 = line1[3] - line1[1];
	double dx2 = line2[2] - line2[0];
	double dy2 = line2[3] - line2[1];

	// Вычисляем углы линий относительно оси X
	double theta1 = std::atan2(dy1, dx1);
	double theta2 = std::atan2(dy2, dx2);

	// Нормализация углов к диапазону [0, 2*PI]
	theta1 = std::fmod(theta1 + 2 * CV_PI, 2 * CV_PI);
	theta2 = std::fmod(theta2 + 2 * CV_PI, 2 * CV_PI);

	// Вычисляем разность углов и приводим результат в диапазон [0, 2*PI]
	double angleDifference = std::fabs(theta1 - theta2);
	angleDifference = std::fmod(angleDifference, 2 * CV_PI);

	// Находим наименьший угол между двумя линиями
	double angleBetweenLines = std::fmin(angleDifference, 2 * CV_PI - angleDifference);

	return angleBetweenLines;
}

int findIntersectionPoint(cv::Vec4f line1, cv::Vec4f line2, cv::Point* crossPoint) {
	double x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
	double x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

	// Вычисляем угловые коэффициенты и свободные члены уравнений прямых
	double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

	if (std::fabs(denom) < 1e-2) {
		return -1; // Прямые параллельны или совпадают
	}

	double intersectX = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
	double intersectY = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

	*crossPoint = cv::Point(cvRound(intersectX), cvRound(intersectY));
	return 0; // Точка пересечения найдена
}

void drawline(cv::Mat img, cv::Vec4f line) {
	cv::Vec4i l = line;
	cv::line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 255, 255), 1, cv::LINE_8);
}

void drawline(cv::Mat img, cv::Vec2f line) {
	float rho = line[0], theta = line[1];
	cv::Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	cv::line(img, pt1, pt2, cv::Scalar(255, 255, 255), 1, cv::LINE_8);
}