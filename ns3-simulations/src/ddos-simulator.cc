#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ipv4-l3-protocol.h" // Cần thiết cho Mitigation
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSSimulator");

// Cấu hình đường dẫn tuyệt đối (Sửa lại nếu cần thiết)
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

    // Realtime Stats
    std::ofstream m_statsFile;
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime;

    // Live Stats & Mitigation
    std::ofstream m_liveStatsFile;
    std::set<Ipv4Address> m_blockedIps;
    std::map<FlowId, uint32_t> m_lastTxPackets_live;

    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    
    void SetupFlowMonitor();
    void ScheduleRealtimeStats();
    void SetupMitigation();
    void SetupNetAnim();

    void UpdateRealtimeStats();
    void MonitorLiveFlows();
    void CheckForBlacklistUpdates();
    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);

    void SaveFinalResults();
};

// ... (Constructor và các hàm Setup cơ bản giữ nguyên) ...
DDoSSimulator::DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime)
    : m_nIotNodes(nIotNodes), m_nAttackers(nAttackers), m_simTime(simTime) {}

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
    NS_LOG_INFO("Created " << m_nIotNodes << " IoT nodes, " << m_nAttackers << " attackers.");
}

void DDoSSimulator::SetupMobility() {
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator", "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0), "DeltaX", DoubleValue(5.0), "DeltaY", DoubleValue(5.0), "GridWidth", UintegerValue(10), "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel", "Bounds", RectangleValue(Rectangle(0, 100, 0, 100)));
    mobility.Install(m_iotNodes);

    Ptr<ListPositionAllocator> staticPos = CreateObject<ListPositionAllocator>();
    staticPos->Add(Vector(25.0, 50.0, 0.0));
    staticPos->Add(Vector(75.0, 50.0, 0.0));
    staticPos->Add(Vector(50.0, 50.0, 0.0));
    MobilityHelper staticMob;
    staticMob.SetPositionAllocator(staticPos);
    staticMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    staticMob.Install(m_baseStations);
    staticMob.Install(m_serverNode);
}

void DDoSSimulator::SetupNetwork() {
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    WifiMacHelper wifiMac;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());

    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);

    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    NetDeviceContainer staDevices = wifi.Install(wifiPhy, wifiMac, m_iotNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer p2p1 = p2p.Install(m_baseStations.Get(0), m_serverNode.Get(0));
    NetDeviceContainer p2p2 = p2p.Install(m_baseStations.Get(1), m_serverNode.Get(0));

    InternetStackHelper stack;
    stack.Install(m_allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_staInterfaces = address.Assign(staDevices);
    address.Assign(m_apDevices);

    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer i2 = address.Assign(p2p1);
    m_serverIp = i2.GetAddress(1);

    address.SetBase("10.1.3.0", "255.255.255.0");
    address.Assign(p2p2);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DDoSSimulator::SetupApplications() {
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 8080));
    ApplicationContainer sinkApp = sinkHelper.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(m_simTime));

    for (uint32_t i = 0; i < m_nIotNodes; ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end()) continue;
        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, 8080));
        onOff.SetAttribute("DataRate", StringValue("50kbps"));
        onOff.SetAttribute("PacketSize", UintegerValue(128));
        onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
        ApplicationContainer app = onOff.Install(m_iotNodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.1));
        app.Stop(Seconds(m_simTime - 1.0));
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
        app.Start(Seconds(5.0));
        app.Stop(Seconds(m_simTime - 5.0));
        
        m_attackerIps.push_back(m_staInterfaces.GetAddress(idx));
    }
}

void DDoSSimulator::SetupNetAnim() {
    static AnimationInterface anim("ddos-animation.xml");
    for(uint32_t i=0; i<m_nIotNodes; ++i) {
         anim.UpdateNodeColor(m_iotNodes.Get(i), 0, 255, 0);
    }
    for(auto idx : m_attackerIndices) {
         anim.UpdateNodeColor(m_iotNodes.Get(idx), 255, 0, 0);
    }
}

void DDoSSimulator::SetupFlowMonitor() {
    m_monitor = m_flowMonHelper.InstallAll();
}

void DDoSSimulator::ScheduleRealtimeStats() {
    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";
    m_statsFile.close();

    for (double t = 1.0; t <= m_simTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &DDoSSimulator::UpdateRealtimeStats, this);
    }
}

void DDoSSimulator::UpdateRealtimeStats() {
    if (!m_monitor) return;

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    uint32_t normalPkts = 0, attackPkts = 0;
    double totalThroughput = 0, totalDelay = 0;
    uint32_t flowCount = 0;

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;

        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        bool isAttack = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

        uint32_t now = it->second.txPackets;
        uint32_t last = m_lastTxPackets_realtime[it->first];
        uint32_t delta = (now > last) ? now - last : 0;

        if (isAttack) attackPkts += delta;
        else normalPkts += delta;

        m_lastTxPackets_realtime[it->first] = now;

        totalThroughput += it->second.rxBytes * 8.0 / 1000.0;
        totalDelay += it->second.delaySum.GetSeconds();
        flowCount++;
    }
    
    double avgDelay = flowCount > 0 ? totalDelay / flowCount : 0;
    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::app);
    m_statsFile << Simulator::Now().GetSeconds() << "," << normalPkts << "," << attackPkts << "," << totalThroughput << "," << avgDelay << "\n";
    m_statsFile.close();
}

// ============================================================================
// CÁC HÀM MITIGATION (ĐÃ ĐƯỢC SỬA LỖI)
// ============================================================================

void DDoSSimulator::SetupMitigation()
{
    std::cout << "MITIGATION (std::cout): Setting up Mitigation..." << std::endl;

    // 1. SỬA: Cài đặt callback cho thiết bị (BẮT BUỘC)
    for (uint32_t i = 0; i < m_apDevices.GetN(); ++i) {
        m_apDevices.Get(i)->SetReceiveCallback(MakeCallback(&DDoSSimulator::PacketDropCallback, this));
    }

    // 2. Khởi tạo file live stats
    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,"
                    << "delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                    << "throughput,flow_duration,label\n";
    m_liveStatsFile.close();

    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

// 3. SỬA: Logic chặn gói tin và chuyển tiếp
bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    if (protocol == 0x0800) { // IPv4
        Ptr<Packet> p_copy = packet->Copy();
        Ipv4Header header;
        if (p_copy->PeekHeader(header)) {
            Ipv4Address srcIp = header.GetSource();
            // Kiểm tra blacklist
            if (m_blockedIps.count(srcIp)) {
                std::cout << "MITIGATION: Dropped packet from blocked IP: " << srcIp << std::endl;
                return true; // HỦY (Drop)
            }
        }
        // Chuyển tiếp gói tin sạch lên lớp IPv4
        Ptr<Ipv4L3Protocol> ipv4l3 = device->GetNode()->GetObject<Ipv4L3Protocol>();
        if (ipv4l3) {
            ipv4l3->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST);
            return true; // CHUYỂN TIẾP (Pass)
        }
    }
    else if (protocol == 0x0806) { // ARP (Rất quan trọng!)
         // Phải cho phép ARP đi qua để mạng hoạt động
         return false; // Trả về false để NS-3 tự xử lý (hoặc xử lý thủ công nếu cần)
    }
    return false;
}

// 4. SỬA: Logic ghi file cho AI
void DDoSSimulator::MonitorLiveFlows()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;
    
    if (!m_monitor) {
        Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
        return;
    }

    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::app);

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;

        uint32_t now = it->second.txPackets;
        uint32_t last = m_lastTxPackets_live[it->first];

        if (now > last) { // Chỉ ghi nếu có cập nhật
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
            
            double flowDuration = it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds();
            double throughput = (flowDuration > 0) ? it->second.rxBytes * 8.0 / (flowDuration * 1000.0) : 0;
            double packetLossRatio = (it->second.txPackets + it->second.lostPackets > 0) ? 
                (double)it->second.lostPackets / (it->second.txPackets + it->second.lostPackets) : 0;
            
            bool isAttack = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

            m_liveStatsFile << Simulator::Now().GetSeconds() << ","
                            << t.sourceAddress << "," << (int)t.protocol << ","
                            << it->second.txPackets << "," << it->second.rxPackets << ","
                            << it->second.txBytes << "," << it->second.rxBytes << ","
                            << it->second.delaySum.GetSeconds() << "," << it->second.jitterSum.GetSeconds() << ","
                            << it->second.lostPackets << "," << packetLossRatio << ","
                            << throughput << "," << flowDuration << "," << (isAttack ? 1 : 0) << "\n";
            
            m_lastTxPackets_live[it->first] = now;
        }
    }
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

// 5. SỬA: Logic đọc Blacklist
void DDoSSimulator::CheckForBlacklistUpdates()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;

    std::string path = PROJECT_ROOT + "/data/live/blacklist.txt";
    std::ifstream blacklistFile(path.c_str());
    if (blacklistFile.is_open()) {
        std::string ipString;
        while (std::getline(blacklistFile, ipString)) {
            if (ipString.empty()) continue;
            Ipv4Address ip;
            ip.Set(ipString.c_str());
            if (m_blockedIps.insert(ip).second) {
                std::cout << "MITIGATION: Adding IP to blacklist: " << ip << std::endl;
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

    std::string path = PROJECT_ROOT + "/data/raw/ns3_detailed_results.csv";
    std::ofstream results(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    results << "flow_id,source_ip,destination_ip,protocol,tx_packets,rx_packets,"
            << "tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
            << "throughput,flow_duration,label\n";

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        double dur = it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds();
        double thr = (dur > 0) ? it->second.rxBytes * 8.0 / (dur * 1000.0) : 0;
        double plr = (it->second.txPackets + it->second.lostPackets > 0) ? 
            (double)it->second.lostPackets / (it->second.txPackets + it->second.lostPackets) : 0;
        bool isAttack = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

        results << it->first << "," << t.sourceAddress << "," << t.destinationAddress << "," << (int)t.protocol << ","
                << it->second.txPackets << "," << it->second.rxPackets << ","
                << it->second.txBytes << "," << it->second.rxBytes << ","
                << it->second.delaySum.GetSeconds() << "," << it->second.jitterSum.GetSeconds() << ","
                << it->second.lostPackets << "," << plr << ","
                << thr << "," << dur << "," << (isAttack ? 1 : 0) << "\n";
    }
    results.close();
}

void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();
    SetupNetAnim();
    SetupFlowMonitor();
    ScheduleRealtimeStats();
    SetupMitigation();

    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();
    SaveFinalResults();
    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    uint32_t nodes = 40;
    uint32_t attackers = 10;
    double simTime = 60.0;

    CommandLine cmd;
    cmd.AddValue("nodes", "Number of normal IoT nodes", nodes);
    cmd.AddValue("attackers", "Number of attacker nodes", attackers);
    cmd.AddValue("time", "Simulation time in seconds", simTime);
    cmd.Parse(argc, argv);

    DDoSSimulator sim(nodes, attackers, simTime);
    sim.Run();

    return 0;
}