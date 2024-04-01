#include <opencv2/opencv.hpp>
#include <iostream>

class PartRectangle {
public:							//	._______.										
	cv::Vec4i parallelLine1;			//	.		.
	cv::Vec4i parallelLine2;			//	|		|
	cv::Vec4i perpendicularLine;			//	|		|
							//	.		.

	PartRectangle(const cv::Vec4i& l1, const cv::Vec4i& l2, const cv::Vec4i& l3)
		: parallelLine1(l1), parallelLine2(l2), perpendicularLine(l3) {
	}
};

class Rectangle
{
public:
	cv::Vec4i line1;
	cv::Vec4i line2;
	cv::Vec4i line3;
	cv::Vec4i line4;


	Rectangle(const cv::Vec4i& l1, const cv::Vec4i& l2, const cv::Vec4i& l3, const cv::Vec4i& l4)
		:line1(l1), line2(l2), line3(l3), line4(l4) {

	}
};

bool isRectangle(PartRectangle condidat1, PartRectangle condidat2);
int  findIntersectionPoint(cv::Vec4f line1, cv::Vec4f line2, cv::Point* crossPoint);
void drawline(cv::Mat img, cv::Vec4f line);
double calculateAngleBetweenLines(cv::Vec4f line1, cv::Vec4f line2);
int isIntersection(cv::Vec4i line1, cv::Vec4i line2);
double approximateDistanceBetweenAlmostParallelLines(cv::Vec4f line1, cv::Vec4f line2);
double distanceBetweenParallelSegments(cv::Vec4f segment1, cv::Vec4f segment2);
double pointToLineDistance(double x, double y, double A, double B, double C);
bool isSegmentBetweenPoints(cv::Vec4i segment, cv::Point pt1, cv::Point pt2);
int orientation(cv::Point pt, cv::Vec4i line);
int distanceSquared(const cv::Point& p1, const cv::Point& p2);
bool areParallelBetweenPerpendiculars(const cv::Vec4i& parallel1, const cv::Vec4i& parallel2,
	const cv::Vec4i& perpendicular1, const cv::Vec4i& perpendicular2);
double segmentLength(const cv::Vec4i& segment);

void filterLines(std::vector<cv::Vec4i>& lines, cv::Mat& img);

void findFeaturePoints(std::vector<cv::Vec4i>& lines, cv::Mat& img, cv::Mat& cany_res, std::vector<cv::Point>& feature_points, std::vector<cv::Vec4i>& lines1_for_point, std::vector<cv::Vec4i>& lines2_for_point);




void findPartRectangle(std::vector<cv::Point>& feature_points, std::vector<cv::Vec4i>& lines1_for_point, std::vector<cv::Vec4i>& lines2_for_point, std::vector<PartRectangle>& condidates);

void findCards(std::vector<PartRectangle>& condidates, std::vector<Rectangle>& cards);

int main(int argc, char** argv) {
	// Загрузка изображения
	
	cv::Mat img_original = cv::imread("../im1.jpg");//argv[1] 
	cv::Mat img = img_original.clone();
	if (img.empty()) {
		std::cout << "Image Not Found!!!" << std::endl;
		return -1;
	}

	double scale_factor = 0.2;
	int new_height = cvRound(img.rows * scale_factor);
	int new_width = cvRound(img.cols * scale_factor);
	cv::resize(img, img, cv::Size(new_width, new_height));
	cv::resize(img_original, img_original, cv::Size(new_width, new_height));

	// Уменьшение шума
	GaussianBlur(img, img, cv::Size(7, 7), 1.5);

	// Конвертация в градации серого
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// Применение детектора границ Canny
	cv::Mat cany_res;
	Canny(gray, cany_res, 50, 150);

	// Вероятностный Хофф
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(cany_res, lines, 2, CV_PI / 180, 75, 30, 15);

	filterLines(lines, img);

	std::vector<cv::Vec4i> lines1_for_point;
	std::vector<cv::Vec4i> lines2_for_point;
	std::vector<cv::Point> feature_points;
	// ищем точки персечения прямых, которые пересекаются под прямым углом 
	findFeaturePoints(lines, img, cany_res, feature_points, lines1_for_point, lines2_for_point);


	std::vector < PartRectangle > condidates;
	findPartRectangle(feature_points, lines1_for_point, lines2_for_point, condidates);


	std::vector<Rectangle> cards;

	//выберем прямоугольники прямоугольники.
	findCards(condidates, cards);


	//Рисуем карточки 
	for (int i = 0; i < cards.size(); i++)
	{

		drawline(img_original, cards[i].line1);
		drawline(img_original, cards[i].line2);
		drawline(img_original, cards[i].line3);
		drawline(img_original, cards[i].line4);
	}
	cv::imshow("RESULT", img_original);
	cv::waitKey(0);
	//вернем к исходному размеру и сохраним.
	scale_factor = 1 / 0.2;
	new_height = cvRound(img.rows * scale_factor);
	new_width = cvRound(img.cols * scale_factor);
	cv::resize(img_original, img_original, cv::Size(new_width, new_height));
	cv::imwrite("result.jpg", img_original);


	return 0;
}

void findCards(std::vector<PartRectangle>& condidates, std::vector<Rectangle>& cards)
{
	for (int i = 0; i < condidates.size(); i++)
	{
		for (int j = i + 1; j < condidates.size(); j++) {
			if (isRectangle(condidates[i], condidates[j])) {
				cv::Vec4i line1(condidates[i].perpendicularLine);
				cv::Vec4i line2;
				cv::Vec4i line3(condidates[j].perpendicularLine);
				cv::Vec4i line4;

				cv::Point pt1(condidates[i].perpendicularLine[0], condidates[i].perpendicularLine[1]);
				cv::Point pt2(condidates[i].perpendicularLine[2], condidates[i].perpendicularLine[3]);
				cv::Point pt3(condidates[j].perpendicularLine[0], condidates[j].perpendicularLine[1]);
				cv::Point pt4(condidates[j].perpendicularLine[2], condidates[j].perpendicularLine[3]);

				int dist1 = distanceSquared(pt1, pt3);
				int dist2 = distanceSquared(pt1, pt4);
				if (dist1 >= dist2)
					line2 = cv::Vec4i(pt1.x, pt1.y, pt4.x, pt4.y);
				else if (dist1 < dist2)
					line2 = cv::Vec4i(pt1.x, pt1.y, pt3.x, pt3.y);

				dist1 = distanceSquared(pt2, pt3);
				dist2 = distanceSquared(pt2, pt4);

				if (dist1 >= dist2)
					line4 = cv::Vec4i(pt2.x, pt2.y, pt4.x, pt4.y);
				else if (dist1 < dist2)
					line4 = cv::Vec4i(pt2.x, pt2.y, pt3.x, pt3.y);


				Rectangle rect(line1, line2, line3, line4);
				double ratio = segmentLength(line1) / segmentLength(line2);
				if ((ratio > 1.5 && ratio < 1.62) || (ratio > 0.61 && ratio < 0.67))
				{
					bool card_eqaul = false;
					for (int i = 0; i < cards.size(); i++)
					{
						card_eqaul = cards[i].line1 == line1 || cards[i].line1 == line2
							|| cards[i].line1 == line3 || cards[i].line1 == line4;
						if (card_eqaul)
							break;
					}
					if (!card_eqaul)
						cards.push_back(rect);
				}

			}
		}
	}
}

void findPartRectangle(std::vector<cv::Point>& feature_points, std::vector<cv::Vec4i>& lines1_for_point, std::vector<cv::Vec4i>& lines2_for_point, std::vector<PartRectangle>& condidates)
{
	for (size_t i = 0; i < feature_points.size(); ++i) {
		for (size_t j = i + 1; j < feature_points.size(); ++j) {

			std::vector<cv::Vec4i> generatingLines = {
				lines1_for_point[i], lines2_for_point[i], lines1_for_point[j],lines2_for_point[j] };
			std::vector<int> number_line = { 0,1,2,3 }; // индексы generatingLines.

			// есть ли среди образующих две точки общие прямых.
			bool foundEqual = false;
			for (size_t k = 0; k < generatingLines.size() && !foundEqual; ++k) {
				for (size_t l = k + 1; l < generatingLines.size(); ++l) {
					if (generatingLines[k] == generatingLines[l]) { // находим два угла с общей прямой 
						foundEqual = true;

						number_line.erase(number_line.begin() + std::max(k, l));
						number_line.erase(number_line.begin() + std::min(k, l));

						// проверим лежит ли образующий отрезок между точками. (У карточки обязательно лежит )
						if (isSegmentBetweenPoints(generatingLines[k], feature_points[i], feature_points[j]))
						{
							int orient1 = orientation(feature_points[i], generatingLines[number_line[0]]);
							int orient2 = orientation(feature_points[j], generatingLines[number_line[1]]);

							if (orient1 == orient2) // не совпадающие прямые должны лежать по одну сторону от точек
							{
								cv::Vec4i parallelLine1 = generatingLines[number_line[0]];
								cv::Vec4i parallelLine2 = generatingLines[number_line[1]];
								cv::Vec4i perpendicularLine(feature_points[i].x, feature_points[i].y, feature_points[j].x, feature_points[j].y);

								PartRectangle part_rect1(parallelLine1, parallelLine2, perpendicularLine);
								condidates.push_back(part_rect1);
							}
							break;
						}
						break;
					}
				}
			}
		}
	}
}

void findFeaturePoints(std::vector<cv::Vec4i>& lines, cv::Mat& img, cv::Mat& cany_res, //input
	std::vector<cv::Point>& feature_points, std::vector<cv::Vec4i>& lines1_for_point, std::vector<cv::Vec4i>& lines2_for_point) //output
{
	double angle_tolerances = 3; //grad 
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
			if (angleBetweenLines > CV_PI / 2 + angle_tolerances * (CV_PI / 180) || angleBetweenLines < CV_PI / 2 - angle_tolerances * (CV_PI / 180))
			{
				continue;
		    }

			// Значение взвешенной суммы в окрестности 3x3 вокруг crossPoint
			// для поиска потенциальных скругленных углов(карточки)
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
				feature_points.push_back(crossPoint);
				lines1_for_point.push_back(lines[i]);
				lines2_for_point.push_back(lines[j]);
			}
		}
	}
}

void filterLines(std::vector<cv::Vec4i>& lines, cv::Mat& img)
{
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
}

//-------------------------------------------------------------------------
// Вспомогательные функции.
double segmentLength(const cv::Vec4i& segment) {
	int x1 = segment[0];
	int y1 = segment[1];
	int x2 = segment[2];
	int y2 = segment[3];

	return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}
bool isRectangle(PartRectangle condidat1, PartRectangle condidat2) {

	bool perpend_different = (condidat1.perpendicularLine != condidat2.perpendicularLine);
	bool paralel_line_equal = (condidat1.parallelLine1 == condidat2.parallelLine1 && condidat1.parallelLine2 == condidat2.parallelLine2)
		|| (condidat1.parallelLine1 == condidat2.parallelLine2 && condidat1.parallelLine2 == condidat2.parallelLine1);

	bool orient = areParallelBetweenPerpendiculars(condidat1.parallelLine1, condidat1.parallelLine2,
		condidat1.perpendicularLine, condidat2.perpendicularLine);

	return perpend_different && paralel_line_equal && orient;
}


bool areParallelBetweenPerpendiculars(const cv::Vec4i& parallel1, const cv::Vec4i& parallel2,
	const cv::Vec4i& perpendicular1, const cv::Vec4i& perpendicular2) {
	// Преобразование координат в точки для параллельных отрезков
	cv::Point p1Start(parallel1[0], parallel1[1]), p1End(parallel1[2], parallel1[3]);
	cv::Point p2Start(parallel2[0], parallel2[1]), p2End(parallel2[2], parallel2[3]);

	// Преобразование координат в точки для перпендикулярных отрезков
	cv::Point perp1Start(perpendicular1[0], perpendicular1[1]), perp1End(perpendicular1[2], perpendicular1[3]);
	cv::Point perp2Start(perpendicular2[0], perpendicular2[1]), perp2End(perpendicular2[2], perpendicular2[3]);

	// Находим ограничивающие координаты для перпендикулярных отрезков
	int minX = std::min({ perp1Start.x, perp1End.x, perp2Start.x, perp2End.x });
	int maxX = std::max({ perp1Start.x, perp1End.x, perp2Start.x, perp2End.x });
	int minY = std::min({ perp1Start.y, perp1End.y, perp2Start.y, perp2End.y });
	int maxY = std::max({ perp1Start.y, perp1End.y, perp2Start.y, perp2End.y });

	// Проверяем, лежат ли параллельные отрезки внутри ограничивающих координат
	return (std::min(p1Start.x, p1End.x) >= minX && std::max(p1Start.x, p1End.x) <= maxX &&
		std::min(p1Start.y, p1End.y) >= minY && std::max(p1Start.y, p1End.y) <= maxY) &&
		(std::min(p2Start.x, p2End.x) >= minX && std::max(p2Start.x, p2End.x) <= maxX &&
			std::min(p2Start.y, p2End.y) >= minY && std::max(p2Start.y, p2End.y) <= maxY);
}


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

	bool isRectangle = (dists[0] - dists[1] < 1) && (dists[2] - dists[3] < 1) && (dists[4] - dists[5] < 1) && (dists[0] - dists[4] < 1);
	if (!isRectangle) return false;

	// Вычисляем соотношение сторон прямоугольника
	double sideRatio = std::sqrt(static_cast<double>(dists[0]) / dists[2]);
	std::cout << "sideRatio = " << sideRatio << std::endl;
	if ((sideRatio > 0.62 && sideRatio < 0.66))
		return true;
	return false;
}

int orientation(cv::Point pt, cv::Vec4i line) {
	cv::Point
		segLeft(std::min(line[0], line[2]), std::min(line[1], line[3])),
		segRight(std::max(line[0], line[2]), std::max(line[1], line[3]));

	if (segRight.x - segLeft.x <= 2) // прямая вертикальная
	{
		if (segRight.y <= pt.y) // прямая левее точки.
			return -1;

		else if (segLeft.y >= pt.y) // прямая правее точки
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
