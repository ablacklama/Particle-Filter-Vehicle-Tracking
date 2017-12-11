/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	

	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}

	is_initialized = true;
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double theta;


	for (vector<Particle>::iterator i = particles.begin(); i != particles.end(); i++) {
		if (fabs(yaw_rate) > 0.0001) {
			theta = i->theta;
			i->x = i->x + ((velocity / yaw_rate)*(sin(theta + (yaw_rate * delta_t)) - sin(theta)));
			i->y = i->y + ((velocity / yaw_rate)*(cos(theta) - cos(theta + (yaw_rate * delta_t))));
			i->theta = theta + (yaw_rate * delta_t);
		}
		else {
			i->x += velocity * delta_t * cos(i->theta);
			i->y += velocity * delta_t * sin(i->theta);
		}
		normal_distribution<double> dist_x(i->x, std_pos[0]);
		normal_distribution<double> dist_y(i->y, std_pos[1]);
		normal_distribution<double> dist_theta(i->theta, std_pos[2]);
	
		i->x = dist_x(gen);
		i->y = dist_y(gen);
		i->theta = dist_theta(gen);
	}
	return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double max_distance = 1.0e+99;

	for (LandmarkObs &obs : observations) {
		double min_distance = max_distance;

		for (auto& pred : predicted) {
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if (distance < min_distance) {
				obs.id = pred.id;
				min_distance = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//  Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weight_sum = 0.0;

	for (auto& particle : particles) {
		vector<LandmarkObs> observeCopy = observations;
		
		//transform each observation to map coordinates relative to this particle
		for (auto& obs : observeCopy) {
			double x = particle.x + (cos(particle.theta)* obs.x) - (sin(particle.theta) * obs.y);
			double y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
			obs.x = x;
			obs.y = y;
		}

		//find the landmarks within the sensor range
		vector<LandmarkObs> predicted;
		for (auto& landmark : map_landmarks.landmark_list) {
			LandmarkObs temp;
			double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
			if (distance <= sensor_range) {
				temp.id = landmark.id_i;
				temp.x = landmark.x_f;
				temp.y = landmark.y_f;
				predicted.push_back(temp);
			}
		}
		

		//set associations for each observation
		dataAssociation(predicted, observeCopy);

		particle.weight = 1.0;
		for (auto& obs : observeCopy) {
			LandmarkObs lm;
			lm.id = obs.id;

			//find the right landmark
			for (auto& landmark : map_landmarks.landmark_list) {
				if (landmark.id_i == lm.id) {
					lm.x = landmark.x_f;
					lm.y = landmark.y_f;
				}
			}
			double t1 = (pow((obs.x - lm.x),2)) / (2 * pow(std_landmark[0],2));
			double t2 = (pow((obs.y - lm.y), 2)) / (2 * pow(std_landmark[1],2));
			particle.weight *= (1 / (2 * M_PI* std_landmark[0] * std_landmark[1])) * exp(-(t1 + t2));
		}
		weight_sum += particle.weight;
		

	}
	vector<double> all_weights;
	for (Particle &p : particles) {
		p.weight /= weight_sum;
		all_weights.push_back(p.weight);
		
	}
	weights = all_weights;
	
}

void ParticleFilter::resample() {
	//  Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	new_particles.resize(num_particles);

	discrete_distribution<int> d(weights.begin(), weights.end());
	

	for (int i = 0; i < num_particles; i++) {
		int j = d(gen);
		new_particles[i] = particles[j];
		new_particles[i].id = i;
	}

	particles = new_particles;
	return;
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
