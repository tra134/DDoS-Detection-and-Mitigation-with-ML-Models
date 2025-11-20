/*
 * DDoS Simulator - DYNAMIC VISUAL FX VERSION
 * - Visual Logic:
 * 1. NORMAL:   Blue   (Traffic < Threshold)
 * 2. ATTACKED: Red    (Attack Traffic High & No Drop)
 * 3. DEFENSE:  Green  (Attack Traffic High & Drops happening)
 * - Topology: 2 Base Stations acting as a unified firewall.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/arp-l3-protocol.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <map> 
#include <set> 
#include <iostream> 
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSSimulator");

// Đường dẫn gốc
const std::string PROJECT_ROOT = "/home/traphan/ns-3-dev/ddos-project-new";

class DDoSSimulator
{
public:
    DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime);
    void Run();

private:
    uint32_t m_nIotNodes;
    uint32_t m_nAttackers;
    double m_simTime;

    NodeContainer m_iotNodes;
    NodeContainer m_baseStations;
    NodeContainer m_serverNode;
    NodeContainer m_allNodes;

    Ipv4InterfaceContainer m_staInterfaces;
    Ipv4Address m_serverIp;
    NetDeviceContainer m_apDevices; 

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    std::ofstream m_statsFile; 
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime; 
    std::ofstream m_liveStatsFile; 
    std::set<Ipv4Address> m_blockedIps; 
    std::map<FlowId, uint32_t> m_lastTxPackets_live;
    
    // Biến đếm số gói tin bị chặn trong 1 giây qua (để đổi màu)
    uint32_t m_droppedPacketsInterval; 
    
    AnimationInterface* m_anim;

    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    
    void SetupFlowMonitor();
    void ScheduleRealtimeStats();
    void SetupMitigation();

    void MonitorLiveFlows();          
    void CheckForBlacklistUpdates();  
    void UpdateRealtimeStats();       
    void SaveFinalResults();
    
    // Hàm xử lý hiệu ứng hình ảnh
    void UpdateVisualEffects(uint32_t attackPackets, uint32_t droppedPackets);

    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);
};

DDoSSimulator::DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime)
{
    m_nIotNodes = nIotNodes;
    m_nAttackers = nAttackers;
    m_simTime = simTime;
    m_anim = 0;
    m_droppedPacketsInterval = 0;
}

// ... (CreateNodes, SetupMobility, SetupNetwork, SetupApplications, SetupDDoSAttack giữ nguyên như bản trước) ...
// ĐỂ TIẾT KIỆM DÒNG, TÔI SẼ VIẾT LẠI CÁC HÀM SETUP CHÍNH, CÁC HÀM CƠ BẢN BẠN CÓ THỂ COPY TỪ BẢN CŨ
// HOẶC DÙNG ĐOẠN CODE ĐẦY ĐỦ DƯỚI ĐÂY:

void DDoSSimulator::CreateNodes() {
    m_iotNodes.Create(m_nIotNodes);
    m_baseStations.Create(2);
    m_serverNode.Create(1);
    m_allNodes.Add(m_iotNodes);
    m_allNodes.Add(m_baseStations);
    m_allNodes.Add(m_serverNode);
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    while (m_attackerIndices.size() < m_nAttackers) {
        uint32_t idx = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), idx) == m_attackerIndices.end())
            m_attackerIndices.push_back(idx);
    }
    std::cout << "✅ TOPOLOGY INITIALIZED: " << m_nIotNodes << " IoT, " << m_nAttackers << " Attackers." << std::endl;
}

void DDoSSimulator::SetupMobility() {
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(20.0), "DeltaY", DoubleValue(20.0),
                                  "GridWidth", UintegerValue(5), "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds", RectangleValue(Rectangle(-10, 200, -10, 200)), "Distance", DoubleValue(5.0));
    mobility.Install(m_iotNodes);

    Ptr<ListPositionAllocator> staticPos = CreateObject<ListPositionAllocator>();
    staticPos->Add(Vector(80.0, 150.0, 0.0));  // BS 1
    staticPos->Add(Vector(80.0, -50.0, 0.0));  // BS 2
    staticPos->Add(Vector(250.0, 50.0, 0.0));  // Server
    MobilityHelper staticMob;
    staticMob.SetPositionAllocator(staticPos);
    staticMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    staticMob.Install(m_baseStations);
    staticMob.Install(m_serverNode);
}

void DDoSSimulator::SetupNetwork() {
    WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211n);
    WifiMacHelper wifiMac; YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy; wifiPhy.SetChannel(wifiChannel.Create());
    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    NetDeviceContainer staDevices = wifi.Install(wifiPhy, wifiMac, m_iotNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps")); 
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer p2p1 = p2p.Install(m_baseStations.Get(0), m_serverNode.Get(0));
    NetDeviceContainer p2p2 = p2p.Install(m_baseStations.Get(1), m_serverNode.Get(0));

    InternetStackHelper stack; stack.Install(m_allNodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0"); m_staInterfaces = address.Assign(staDevices); address.Assign(m_apDevices); 
    address.SetBase("10.1.2.0", "255.255.255.0"); Ipv4InterfaceContainer i2 = address.Assign(p2p1); m_serverIp = i2.GetAddress(1); 
    address.SetBase("10.1.3.0", "255.255.255.0"); address.Assign(p2p2);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DDoSSimulator::SetupApplications() {
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 8080));
    ApplicationContainer sinkApp = sink.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0)); sinkApp.Stop(Seconds(m_simTime));
    for (uint32_t i = 0; i < m_nIotNodes; ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end()) continue; 
        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, 8080));
        onOff.SetAttribute("DataRate", StringValue("50kbps"));
        onOff.SetAttribute("PacketSize", UintegerValue(512));
        onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
        ApplicationContainer app = onOff.Install(m_iotNodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.2)); app.Stop(Seconds(m_simTime - 1.0));
    }
}

void DDoSSimulator::SetupDDoSAttack() {
    for (auto idx : m_attackerIndices) {
        OnOffHelper attack("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, 8080));
        attack.SetAttribute("DataRate", StringValue("5000kbps"));
        attack.SetAttribute("PacketSize", UintegerValue(1024));
        attack.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
        attack.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer app = attack.Install(m_iotNodes.Get(idx));
        app.Start(Seconds(5.0)); app.Stop(Seconds(m_simTime - 5.0));
        m_attackerIps.push_back(m_staInterfaces.GetAddress(idx));
    }
    std::cout << "⚔️  ATTACK SCENARIO STARTED (5s)" << std::endl;
}

void DDoSSimulator::SetupFlowMonitor() { m_monitor = m_flowMonHelper.InstallAll(); }

void DDoSSimulator::ScheduleRealtimeStats() {
    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc); 
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";
    m_statsFile.close();
    for (double t = 1.0; t <= m_simTime; t += 1.0) Simulator::Schedule(Seconds(t), &DDoSSimulator::UpdateRealtimeStats, this);
}

// =================================================================
// === HÀM CẬP NHẬT VISUAL FX (QUAN TRỌNG) ===
// =================================================================
void DDoSSimulator::UpdateVisualEffects(uint32_t attackPackets, uint32_t droppedPackets)
{
    if (!m_anim) return;

    uint8_t r, g, b;
    std::string label;
    double size;

    // TRẠNG THÁI 1: DEFENSE MODE (Đang chặn thành công)
    // Điều kiện: Có gói tin bị chặn
    if (droppedPackets > 0) {
        r = 0; g = 255; b = 0;       // XANH LÁ (An toàn/Phòng thủ)
        label = "🛡️ MITIGATING";
        size = 4.5;                  // To ra để thể hiện đang hoạt động
    }
    // TRẠNG THÁI 2: UNDER ATTACK (Bị tấn công nhưng chưa chặn)
    // Điều kiện: Có nhiều gói tin tấn công (>100) và KHÔNG có gói nào bị chặn
    else if (attackPackets > 100) {
        r = 255; g = 0; b = 0;       // ĐỎ (Nguy hiểm)
        label = "⚠️ UNDER ATTACK";
        size = 6.0;                  // Rất to để cảnh báo
    }
    // TRẠNG THÁI 3: NORMAL (Bình thường)
    else {
        r = 0; g = 191; b = 255;     // XANH DƯƠNG (Yên bình)
        label = "Gateway";
        size = 3.5;
    }

    // Cập nhật đồng bộ cho cả 2 Base Stations
    for (uint32_t i = 0; i < m_baseStations.GetN(); ++i) {
        m_anim->UpdateNodeColor(m_baseStations.Get(i), r, g, b);
        m_anim->UpdateNodeDescription(m_baseStations.Get(i), label);
        m_anim->UpdateNodeSize(m_baseStations.Get(i), size, size);
    }
}

void DDoSSimulator::UpdateRealtimeStats()
{
    if (!m_monitor) return;
    // m_monitor->CheckForLostPackets(); // Tắt để tối ưu visual realtime

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    
    uint32_t norm = 0, attk = 0;
    double tput = 0;
    
    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;
        tput += it->second.rxBytes * 8.0 / 1000.0;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        bool isAtk = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();
        
        uint32_t now = it->second.txPackets;
        uint32_t last = m_lastTxPackets_realtime[it->first];
        uint32_t delta = (now > last) ? now - last : 0;
        
        if (isAtk) attk += delta; else norm += delta;
        m_lastTxPackets_realtime[it->first] = now;
    }
    
    double now = Simulator::Now().GetSeconds();
    
    // <<< CẬP NHẬT HÌNH ẢNH DỰA TRÊN THÔNG SỐ THỰC TẾ >>>
    // Truyền vào số gói tấn công VÀ số gói bị chặn (m_droppedPacketsInterval)
    UpdateVisualEffects(attk, m_droppedPacketsInterval);
    
    // Reset biến đếm drop cho giây tiếp theo
    m_droppedPacketsInterval = 0;

    if (attk > 0 || norm > 0) {
        std::cout << "STATS: T=" << std::setw(3) << now 
                  << " | Norm=" << std::setw(4) << norm 
                  << " | Attk=" << std::setw(5) << attk << std::endl;
    }

    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::app);
    m_statsFile << now << "," << norm << "," << attk << "," << tput << ",0\n";
    m_statsFile.close();
}

void DDoSSimulator::SetupMitigation()
{
    for (uint32_t i = 0; i < m_apDevices.GetN(); ++i) {
        m_apDevices.Get(i)->SetReceiveCallback(MakeCallback(&DDoSSimulator::PacketDropCallback, this));
    }
    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,throughput,flow_duration,label\n";
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    if (protocol == 0x0800) { 
        Ptr<Packet> p_copy = packet->Copy();
        Ipv4Header header;
        if (p_copy->PeekHeader(header)) {
            if (m_blockedIps.count(header.GetSource())) {
                // <<< TĂNG BIẾN ĐẾM DROP ĐỂ ĐỔI MÀU >>>
                m_droppedPacketsInterval++; 
                return true; // DROP
            }
        }
        Ptr<Ipv4L3Protocol> ipv4l3 = device->GetNode()->GetObject<Ipv4L3Protocol>();
        if (ipv4l3) { ipv4l3->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST); return true; }
    } else if (protocol == 0x0806) {
        Ptr<ArpL3Protocol> arp = device->GetNode()->GetObject<ArpL3Protocol>();
        if (arp) { arp->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST); return true; }
    }
    return false; 
}

void DDoSSimulator::MonitorLiveFlows() {
    if (Simulator::Now().GetSeconds() > m_simTime) return; 
    if (!m_monitor) { Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this); return; }
    
    // m_monitor->CheckForLostPackets(); 
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::app);

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;
        uint32_t now = it->second.txPackets;
        if (now > m_lastTxPackets_live[it->first]) {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
            bool isAtk = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();
            m_liveStatsFile << Simulator::Now().GetSeconds() << "," << t.sourceAddress << "," << (int)t.protocol << ","
                            << it->second.txPackets << "," << it->second.rxPackets << "," << it->second.txBytes << "," << it->second.rxBytes << ","
                            << it->second.delaySum.GetSeconds() << "," << it->second.jitterSum.GetSeconds() << ","
                            << it->second.lostPackets << ",0,0,0," << (isAtk?1:0) << "\n";
            m_lastTxPackets_live[it->first] = now;
        }
    }
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

void DDoSSimulator::CheckForBlacklistUpdates() {
    if (Simulator::Now().GetSeconds() > m_simTime) return;
    std::string path = PROJECT_ROOT + "/data/live/blacklist.txt";
    std::ifstream blacklistFile(path.c_str());
    if (blacklistFile.is_open()) {
        std::string ip;
        while (std::getline(blacklistFile, ip)) {
            if (!ip.empty()) {
                if (m_blockedIps.insert(Ipv4Address(ip.c_str())).second)
                    std::cout << "🚫 BLOCKED: " << ip << std::endl;
            }
        }
        blacklistFile.close();
    }
    Simulator::Schedule(Seconds(0.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

void DDoSSimulator::SaveFinalResults() {
    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    std::ofstream resultsFile;
    std::string path = PROJECT_ROOT + "/data/raw/ns3_detailed_results.csv";
    resultsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc); 
    resultsFile << "flow_id,source_ip,destination_ip,protocol,tx_packets,rx_packets,"
                << "tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                << "throughput,flow_duration,label\n";

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        double dur = it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds();
        double thr = (dur>0)? it->second.rxBytes*8.0/(dur*1000.0) : 0;
        double loss = (it->second.txPackets>0)? (double)it->second.lostPackets/(it->second.txPackets+it->second.lostPackets) : 0;
        bool isAtk = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();
        resultsFile << it->first << "," << t.sourceAddress << "," << t.destinationAddress << "," << (int)t.protocol << ","
                    << it->second.txPackets << "," << it->second.rxPackets << "," << it->second.txBytes << "," << it->second.rxBytes << ","
                    << it->second.delaySum.GetSeconds() << "," << it->second.jitterSum.GetSeconds() << "," << it->second.lostPackets << ","
                    << loss << "," << thr << "," << dur << "," << (isAtk?1:0) << "\n";
    }
    resultsFile.close();
}

void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();
    
    // --- NETANIM FINAL CONFIG ---
    AnimationInterface anim("ddos-animation.xml");
    m_anim = &anim;
    anim.EnablePacketMetadata(false); // Tắt để tránh crash nhưng vẫn có chấm di chuyển
    anim.SetMaxPktsPerTraceFile(300000);

    // Coloring
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        Ptr<Node> n = m_iotNodes.Get(i);
        bool isAtk = std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end();
        anim.UpdateNodeColor(n, isAtk ? 255 : 0, isAtk ? 0 : 255, 127); 
        anim.UpdateNodeDescription(n, isAtk ? "Attacker" : "IoT");
        anim.UpdateNodeSize(n, isAtk ? 1.5 : 1.0, isAtk ? 1.5 : 1.0);
    }
    // Ban đầu BS màu Xanh Dương
    for (uint32_t i = 0; i < m_baseStations.GetN(); ++i) {
        anim.UpdateNodeColor(m_baseStations.Get(i), 0, 191, 255); 
        anim.UpdateNodeSize(m_baseStations.Get(i), 3.5, 3.5);
        anim.UpdateNodeDescription(m_baseStations.Get(i), "Gateway");
    }
    anim.UpdateNodeColor(m_serverNode.Get(0), 128, 0, 128); 
    anim.UpdateNodeSize(m_serverNode.Get(0), 6.0, 6.0);
    anim.UpdateNodeDescription(m_serverNode.Get(0), "SERVER");

    SetupFlowMonitor();
    ScheduleRealtimeStats();
    SetupMitigation(); 

    std::cout << "\n🚀 STARTING SIMULATION..." << std::endl;
    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();
    
    m_anim = 0;
    Simulator::Destroy();
    std::cout << "✅ FINISHED. Animation saved." << std::endl;
}

int main(int argc, char* argv[]) {
    uint32_t n=30, a=10; double t=60.0; 
    CommandLine cmd;
    cmd.AddValue("nodes", "n", n); cmd.AddValue("attackers", "a", a); cmd.AddValue("time", "t", t);
    cmd.Parse(argc, argv);
    DDoSSimulator sim(n, a, t);
    sim.Run();
    return 0;
}