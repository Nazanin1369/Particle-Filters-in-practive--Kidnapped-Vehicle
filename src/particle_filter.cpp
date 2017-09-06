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

using namespace std;
#define MIN_YAW_RATE 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Define number of particles
	num_particles = 42;

	// Define unified random generator
	std::default_random_engine gen;

	// Create normal (Gaussian) distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialized particles and their weights
	for (unsigned int i = 0; i < num_particles; ++i) {
		struct Particle new_particle;

		new_particle.id = i;
		new_particle.x = dist_x(gen);
		new_particle.y = dist_y(gen);
		new_particle.theta = dist_theta(gen);
		new_particle.weight = 1.0;

		// Add the new particle to set of particles
		particles.push_back(new_particle);
		weights.push_back(1.0F);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);


	// Pre-calculate pre-factor for motion step:
	const bool yaw_rate_not_zero = std::fabs(yaw_rate) > MIN_YAW_RATE;
	const double fac = yaw_rate_not_zero ? velocity/yaw_rate : velocity;

	for(unsigned int i = 0; i < num_particles; ++i) {
		Particle& current_particle = particles[i];

		if(yaw_rate_not_zero) {
			const double theta_new   = current_particle.theta + yaw_rate * delta_t;
            const double theta_old   = current_particle.theta;
            const double sinThetaNew = std::sin(theta_new);
            const double sinThetaOld = std::sin(theta_old);
            const double cosThetaNew = std::cos(theta_new);
            const double cosThetaOld = std::cos(theta_old);

            current_particle.x      += fac * (sinThetaNew - sinThetaOld);
            current_particle.y      += fac * (cosThetaOld - cosThetaNew);
            current_particle.theta   = theta_new;
		} else {
			const double sinTheta = std::sin(current_particle.theta);
            const double cosTheta = std::cos(current_particle.theta);

            current_particle.x     += fac * cosTheta * delta_t;
            current_particle.y     += fac * sinTheta * delta_t;
		}

		// Adding noise
		const double noise_x     = dist_x(gen);
        const double noise_y     = dist_y(gen);
        const double noise_theta = dist_theta(gen);

        current_particle.x      += dist_x(gen);
        current_particle.y      += dist_y(gen);
        current_particle.theta  += dist_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	for (int i = 0; i < observations.size(); i++)
    {
        LandmarkObs & measurement = observations[i];
        measurement.id = 0;

        double bestAssociationDistance = sensor_range;
        for (unsigned int j = 0; j < predicted.size(); ++j) {
            const LandmarkObs & prediction = predicted[j];

            double distance = dist(measurement.x, measurement.y, prediction.x, prediction.y);
            if (distance < bestAssociationDistance) {
                measurement.id = prediction.id;
                bestAssociationDistance = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

		for (int i = 0; i < num_particles; i++) {
			Particle & particle = particles[i];
			std::vector<LandmarkObs> predictions;
			std::vector<int> associations;
			std::vector<double> sense_x;
			std::vector<double> sense_y;
			double weight = 1;

			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
				const Map::single_landmark_s & landmark = map_landmarks.landmark_list[k];

				// Calculate the offset
				double dx = landmark.x_f - particle.x;
				double dy = landmark.y_f - particle.y;
				double sense_x = dx * cos(particle.theta) + dy * sin(particle.theta);
				double sense_y = -dx * sin(particle.theta) + dy * cos(particle.theta);

				LandmarkObs predicted;
				predicted.id = landmark.id_i;
				predicted.x = sense_x;
				predicted.y = sense_y;

				predictions.push_back(predicted);
			}

			dataAssociation(predictions, observations, sensor_range);

			for (unsigned int j = 0; j < observations.size(); ++j) {
				const LandmarkObs & measurement = observations[j];

				for (unsigned int k = 0; k < predictions.size(); ++k) {
					const LandmarkObs & prediction = predictions[k];

					if (prediction.id == measurement.id) {
						double dx = measurement.x - prediction.x;
						double dy = measurement.y - prediction.y;
						double w_j = exp(-0.5 * (dx * dx / (std_landmark[0] * std_landmark[0]) + dy * dy / (std_landmark[1] * std_landmark[1]))) /
							sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);

						weight *= w_j;

						associations.push_back(measurement.id);
						sense_x.push_back(particle.x + measurement.x * cos(-particle.theta) + measurement.y * sin(-particle.theta));
						sense_y.push_back(particle.y - measurement.x * sin(-particle.theta) + measurement.y * cos(-particle.theta));

						break;
					}
				}
			}

			particle.weight = weight;
			weights[i] = particle.weight;

			particle = SetAssociations(particle, associations, sense_x, sense_y);
		}
}

void ParticleFilter::resample() {
	default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++)
    {
        new_particles.push_back(particles[distribution(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
