#include <vector>
#include <torch/extension.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace bg = boost::geometry;

using point      = boost::geometry::model::d2::point_xy<float>;
using polygon    = boost::geometry::model::polygon<point>;
using linestring = boost::geometry::model::linestring<point>;

#include <iostream>

template <typename T>
int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

int orientation(point a, point b, point r)
{
    // returns one of (-1, 0, 1), -1 means on r is on left of (a -> b), 0 on (a -> b), +1 on right of (a -> b)

    float ax = bg::get<0>(a), ay = bg::get<1>(a);
    float bx = bg::get<0>(b), by = bg::get<1>(b);
    float rx = bg::get<0>(r), ry = bg::get<1>(r);

    float bax = bx - ax;
    float bay = by - ay;

    float rbx = rx - bx;
    float rby = ry - by;

    return sign(bay * rbx - bax * rby);
}

bool on_segment(point a, point b, point r)
{
    float ax = bg::get<0>(a), ay = bg::get<1>(a);
    float bx = bg::get<0>(b), by = bg::get<1>(b);
    float rx = bg::get<0>(r), ry = bg::get<1>(r);

    return (rx <= std::max(ax, bx) && rx >= std::min(ax, bx) &&
            ry <= std::max(ay, by) && ry >= std::min(ay, by));
}

bool segment_intersect(point a, point b, point p, point q)
{
    // return true if (a -> b) intersects with (p -> q), false otherwise

    int o1 = orientation(a, b, p);
    int o2 = orientation(a, b, q);
    int o3 = orientation(p, q, a);
    int o4 = orientation(p, q, b);

    if (o1 != o2 && o3 != o4) return true;

    if (o1 == 0 && on_segment(a, b, p)) return true;
    if (o2 == 0 && on_segment(a, b, q)) return true;
    if (o3 == 0 && on_segment(p, q, a)) return true;
    if (o4 == 0 && on_segment(p, q, b)) return true;

    return false;
}

float segment_distance(point p, point q, point s, point d)
{
    // ASSUMPTION: segment (p -> q) intersects with (s -> s + 1000 * d)
    // Distance to segment (p -> q) in direction d from point s

    float px = bg::get<0>(p), py = bg::get<1>(p);
    float qx = bg::get<0>(q), qy = bg::get<1>(q);
    float sx = bg::get<0>(s), sy = bg::get<1>(s);
    float dx = bg::get<0>(d), dy = bg::get<1>(d);

    // if (!segment_intersect(p, q, s, point{ sx + 1000.f * dx, sy + 1000.f * dy })) {
    //     return std::numeric_limits<float>::infinity();
    // }

    float qpx = qx - px, qpy = qy - py;
    float psx = px - sx, psy = py - sy;

    float proj  = psy * qpx - psx * qpy;
    float denom = dy * qpx - dx * qpy;
    float dist = proj / denom;

    return dist < 0.f ? std::numeric_limits<float>::infinity() : dist;
}


struct RaceTrack
{
    bool is_valid() const
    {
        return !bg::intersects(line);
    }


    polygon             racetrack;  // vertices are stored counter-clockwise
    linestring          line;
    std::vector<point>  left;       // left boundary
    std::vector<point>  right;      // right boundary
    size_t              length;
};


std::vector<RaceTrack> create_racetracks(at::Tensor left, at::Tensor right)
{
    // (float32) left  = [b, s, 2]
    // (float32) right = [b, s, 2]

    size_t b = left.size(0);
    size_t s = left.size(1);

    std::vector<RaceTrack> tracks(b);

    auto left_view  =  left.accessor<float, 3>();
    auto right_view = right.accessor<float, 3>();

    #pragma omp parallel for
    for (size_t i = 0; i < b; ++i) {
        tracks[i].left.resize(s);
        tracks[i].right.resize(s);

        for (size_t j = 1; j <= s; ++j) {
            auto p = point{ left_view[i][s - j][0], left_view[i][s - j][1] };
            bg::append(tracks[i].racetrack.outer(), p);
            bg::append(tracks[i].line, p);
            tracks[i].left[s - j] = p;
        }
        for (size_t j = 0; j < s; ++j) {
            auto p = point{ right_view[i][j][0], right_view[i][j][1] };
            bg::append(tracks[i].racetrack.outer(), p);
            bg::append(tracks[i].line, p);
            tracks[i].right[j] = p;
        }

        tracks[i].length = s - 1;
    }

    return tracks;
}

struct Player
{
    Player(const RaceTrack &track_)
        : track(track_)
    {
    }

    const RaceTrack &track;
    point position = point{ 0.0f, 0.1f };  // WARNING!
    int segment = 0;
};

struct Game
{
    Game(at::Tensor left, at::Tensor right, size_t num_players_)
        : tracks(create_racetracks(left, right))
        , num_players(num_players_)
    {
        // (float32) left   = [b, s, 2]
        // (float32) right  = [b, s, 2]

        size_t b = tracks.size();

        for (size_t i = 0; i < b; ++i) {
            for (size_t j = 0; j < num_players; ++j) {
                players.emplace_back(tracks[i]);
            }
        }
    }

    at::Tensor validate_tracks()
    {
        // (uint8_t) valid = [b]

        auto valid = at::empty({ tracks.size() }, torch::CPU(at::kByte));
        auto valid_view = valid.accessor<uint8_t, 1>();

        #pragma omp parallel for
        for (size_t i = 0; i < tracks.size(); ++i) {
            valid_view[i] = static_cast<uint8_t>(tracks[i].is_valid());
        }

        return valid;
    }

    std::pair<at::Tensor, at::Tensor> update_players(at::Tensor idx, at::Tensor new_positions)
    {
        // (int64_t)    idx = [k]
        // (float32)    new_position = [k, 2]
        // (uint8_t)    dead = [k]
        // (uint8_t)    finished = [k]

        size_t k = idx.size(0);

        auto dead = at::empty({ k }, torch::CPU(at::kByte));
        auto finished = at::empty({ k }, torch::CPU(at::kByte));

        auto idx_view       = idx.accessor<int64_t, 1>();
        auto new_pos_view   = new_positions.accessor<float, 2>();
        auto dead_view      = dead.accessor<uint8_t, 1>();
        auto finished_view  = finished.accessor<uint8_t, 1>();

        #pragma omp parallel for
        for (size_t i = 0; i < k; ++i) {
            auto &player = players[idx_view[i]];
            point new_pos{ new_pos_view[i][0], new_pos_view[i][1] };

            // Compute 'move' segment
            int next_seg = player.segment;

            // forward check
            bool is_alive = true;
            bool is_done = false;

            while (next_seg < player.track.length) {
                auto left_a = player.track.left[next_seg];
                auto left_b = player.track.left[next_seg + 1];

                auto right_a = player.track.right[next_seg];
                auto right_b = player.track.right[next_seg + 1];

                if (segment_intersect( left_a,  left_b, player.position, new_pos) ||
                    segment_intersect(right_a, right_b, player.position, new_pos)) {
                    is_alive = false;  // is dead
                    break;
                }

                int side = orientation(left_b, right_b, new_pos);
                if (side > 0) {  // stayed in the same segment
                    break;
                }
                ++next_seg;
            }

            if (is_alive && next_seg >= player.track.length) {
                // finished race, we can stop here
                is_done = true;
            }

            if (next_seg == player.segment && is_alive && !is_done) {
                while (next_seg >= 0) {
                    auto left_a = player.track.left[next_seg];
                    auto left_b = player.track.left[next_seg + 1];

                    auto right_a = player.track.right[next_seg];
                    auto right_b = player.track.right[next_seg + 1];

                    if (segment_intersect( left_a,  left_b, player.position, new_pos) ||
                        segment_intersect(right_a, right_b, player.position, new_pos)) {
                        is_alive = false;  // is dead
                        break;
                    }

                    int side = orientation(left_a, right_a, new_pos);
                    if (side < 0) {  // stayed in the same segment
                        break;
                    }
                    --next_seg;
                }

                if (next_seg < 0) {
                    is_alive = false;
                }
            }

            player.position = new_pos;
            player.segment = next_seg;

            dead_view[i] = static_cast<uint8_t>(!is_alive);
            finished_view[i] = static_cast<uint8_t>(is_done);
        }

        return { dead, finished };
    }

    at::Tensor smallest_distance(at::Tensor idx, at::Tensor directions)
    {
        /*
        (int64_t)    idx = [k]
        (float32)    directions = [k, d, 4]
        (float32)    distances = [k, d]

        Returns smallest distance in given directions.
        */

        size_t k_size = idx.size(0);
        size_t d_size = directions.size(1);

        auto distances = at::empty({ k_size, d_size }, torch::CPU(at::kFloat));

        auto idx_view  = idx.accessor<int64_t, 1>();
        auto dir_view  = directions.accessor<float, 3>();
        auto dist_view = distances.accessor<float, 2>();

        #pragma omp parallel for
        for (size_t k = 0; k < k_size; ++k) {
            auto &player = players[idx_view[k]];

            for (size_t d = 0; d < d_size; ++d) {
                float x = dir_view[k][d][0], y = dir_view[k][d][1], dx = dir_view[k][d][2], dy = dir_view[k][d][3];

                std::vector<point> out;
                bg::intersection(player.track.line,
                                 linestring{ point{ x, y },
                                             point{ x + 1000.f * dx, y + 1000.f * dy }}, out);

                float min_dist = std::numeric_limits<float>::infinity();
                for (auto &&p : out) {
                    float dist = bg::distance(point{ x, y }, p);
                    min_dist = std::min(min_dist, dist);
                }
                dist_view[k][d] = min_dist;
            }
        }

        return distances;
    }

    std::vector<RaceTrack>  tracks;         // [b]
    std::vector<Player>     players;        // [b * p]
    size_t                  num_players;    // p
};


/*
    Old version
*/


void collision(at::Tensor tracks, at::Tensor segments, at::Tensor output)
{
    /*
    (float32) tracks   = [b, s, 2]
    (float32) segments = [b, p, 4]
    (uint8_t) output   = [b, p]

    Checks if there is a collision between tracks & segments
    */

    size_t b_size = tracks.size(0);
    size_t s_size = tracks.size(1);
    size_t p_size = segments.size(1);

    auto view = tracks.accessor<float, 3>();

    std::vector<linestring> polygons(tracks.size(0));

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t s = 0; s < s_size; ++s) {
            bg::append(polygons[b], point{ view[b][s][0], view[b][s][1] });
        }
    }

    auto seg_view = segments.accessor<float, 3>();
    auto out_view = output.accessor<uint8_t, 2>();

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t p = 0; p < p_size; ++p) {
            bool do_inter = bg::intersects(polygons[b], linestring{ point{ seg_view[b][p][0], seg_view[b][p][1] },
                                                                    point{ seg_view[b][p][2], seg_view[b][p][3] } });
            out_view[b][p] = static_cast<uint8_t>(do_inter);
        }
    }
}

void smallest_distance(at::Tensor tracks, at::Tensor directions, at::Tensor output)
{
    /*
    (float32) tracks     = [b, s, 2]
    (float32) directions = [b, d, 4]
    (float32) output     = [b, d]

    Returns smallest distance in given directions.
    */

    size_t b_size = tracks.size(0);
    size_t s_size = tracks.size(1);
    size_t d_size = directions.size(1);

    auto view = tracks.accessor<float, 3>();

    std::vector<linestring> polygons(tracks.size(0));

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t s = 0; s < s_size; ++s) {
            bg::append(polygons[b], point{ view[b][s][0], view[b][s][1] });
        }
    }

    auto dir_view = directions.accessor<float, 3>();
    auto out_view = output.accessor<float, 2>();

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t d = 0; d < d_size; ++d) {
            float x = dir_view[b][d][0], y = dir_view[b][d][1], dx = dir_view[b][d][2], dy = dir_view[b][d][3];

            std::vector<point> out;
            bg::intersection(polygons[b], linestring{ point{ x, y },
                                                      point{ x + 1000.f * dx, y + 1000.f * dy } }, out);

            float min_dist = std::numeric_limits<float>::infinity();
            for (auto &&p : out) {
                float dist = bg::distance(point{ dir_view[b][d][0], dir_view[b][d][1] }, p);
                min_dist = std::min(min_dist, dist);
            }
            out_view[b][d] = min_dist;
        }
    }
}


void is_valid(at::Tensor tracks, at::Tensor output)
{
    /*
    (float32) tracks     = [b, s, 2]
    (uint8_t) output     = [b]

    Checks if tracks are valid (not self-intersecting).
    */

    size_t b_size = tracks.size(0);
    size_t s_size = tracks.size(1);

    auto view = tracks.accessor<float, 3>();

    std::vector<linestring> polygons(tracks.size(0));

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t s = 0; s < s_size; ++s) {
            bg::append(polygons[b], point{ view[b][s][0], view[b][s][1] });
        }
    }

    auto out_view = output.accessor<uint8_t, 1>();

    #pragma omp parallel for
    for (size_t b = 0; b < b_size; ++b) {
        out_view[b] = !static_cast<uint8_t>(bg::intersects(polygons[b]));
    }
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collision", &collision, "Collision between tracks and segments.");
    m.def("smallest_distance", &smallest_distance, "Smallest distance in given direction.");
    m.def("is_valid", &is_valid, "Checks if tracks are valid (not self-intersecting).");

    py::class_<Game>(m, "Game")
        .def(py::init<at::Tensor, at::Tensor, size_t>())
        .def("validate_tracks", &Game::validate_tracks)
        .def("update_players", &Game::update_players)
        .def("smallest_distance", &Game::smallest_distance);
}






/*

    SHITTY CODE! DO NOT USE! (maybe I will try to make it correct later...)

*/


        // (int64_t)    idx = [k]
        // (float32)    directions = [k, d, 4] -- DIRECTIONS ARE SORTED TO BE CLOCKWISE
        // (float32)    distances = [k, d]

#if 0  // not working...
        size_t k_size = idx.size(0);
        size_t d_size = directions.size(1);

        auto distances = at::empty({ k_size, d_size }, torch::CPU(at::kFloat));

        auto idx_view  = idx.accessor<int64_t, 1>();
        auto dir_view  = directions.accessor<float, 3>();
        auto dist_view = distances.accessor<float, 2>();

        #pragma omp parallel for
        for (size_t k = 0; k < k_size; ++k) {
            auto &player = players[idx_view[k]];

            point  d1{ dir_view[k][0][0], dir_view[k][0][1] };
            point dir{ dir_view[k][0][2], dir_view[k][0][3] };
            point  d2{ bg::get<0>(d1) + 1000.f * bg::get<0>(dir),
                       bg::get<1>(d1) + 1000.f * bg::get<1>(dir) };

            auto &outer = player.track.racetrack.outer();

            // look for first intersection
            size_t first = 0;
            float min_dist = std::numeric_limits<float>::infinity();

            for (size_t i = 0; i < outer.size(); ++i) {
                if (segment_intersect(outer[i], outer[(i + 1) % outer.size()], d1, d2)) {
                    float dist = segment_distance(outer[i], outer[(i + 1) % outer.size()], d1, dir);
                    if (dist < min_dist) {
                        min_dist = dist;
                        first = i;
                    }
                }
            }

            std::vector<float> dists(d_size, std::numeric_limits<float>::infinity());

            // counter-clockwise iteration
            int32_t dir_idx = 0;
            for (size_t i = 0; i <= outer.size(); ++i) {
                size_t j = (first + i) % outer.size();
                if (segment_intersect(outer[j], outer[(j + 1) % outer.size()], d1, d2)) {
                    dists[dir_idx] = segment_distance(outer[j], outer[(j + 1) % outer.size()], d1, dir);
                    --dir_idx;

                    if (dir_idx == 0) break;

                    dir_idx %= d_size;
                    d1  = point{ dir_view[k][dir_idx][0], dir_view[k][dir_idx][1] };
                    dir = point{ dir_view[k][dir_idx][2], dir_view[k][dir_idx][3] };
                    d2  = point{ bg::get<0>(d1) + 1000.f * bg::get<0>(dir),
                                 bg::get<1>(d1) + 1000.f * bg::get<1>(dir) };

                    --i;
                }
            }

            // clockwise iteration
            d1  = point{ dir_view[k][0][0], dir_view[k][0][1] };
            dir = point{ dir_view[k][0][2], dir_view[k][0][3] };
            d2  = point{ bg::get<0>(d1) + 1000.f * bg::get<0>(dir),
                         bg::get<1>(d1) + 1000.f * bg::get<1>(dir) };

            dir_idx = 0;
            for (size_t i = 0; i <= outer.size(); ++i) {
                size_t j = (first - i) % outer.size();
                if (segment_intersect(outer[j], outer[(j + 1) % outer.size()], d1, d2)) {
                    dist_view[k][dir_idx] = std::min(dists[dir_idx],
                                                     segment_distance(outer[j], outer[(j + 1) % outer.size()], d1, dir));
                    ++dir_idx;

                    if (dir_idx >= d_size) break;

                    d1  = point{ dir_view[k][dir_idx][0], dir_view[k][dir_idx][1] };
                    dir = point{ dir_view[k][dir_idx][2], dir_view[k][dir_idx][3] };
                    d2  = point{ bg::get<0>(d1) + 1000.f * bg::get<0>(dir),
                                 bg::get<1>(d1) + 1000.f * bg::get<1>(dir) };

                    --i;
                }
            }
        }

        return distances;
#endif