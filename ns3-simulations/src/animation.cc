// animation_only_gateway_attack.cc
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AnimationGatewayAttack");

class DDoSSimulator
{
public:
    DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime);
    ~DDoSSimulator();
    void Run();

private:
    uint32_t m_nIotNodes;
    uint32_t m_nAttackers;
    double m_simTime;

    NodeContainer m_iotNodes;
    NodeContainer m_baseStations;
    NodeContainer m_serverNode;
    NodeContainer m_allNodes;

    NetDeviceContainer m_apDevices;
    Ipv4InterfaceContainer m_staInterfaces;
    Ipv4Address m_serverIp;

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    AnimationInterface *m_anim = nullptr;

    std::set<Ipv4Address> m_blockedIps;
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime;

    uint32_t m_attackDetectThresholdPacketsPerSec = 50; // giảm để dễ trigger
    double m_mitigationRestoreDelay = 5.0;

    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    void SetupFlowMonitor();
    void SetupNetAnim();
    void MonitorLiveFlows();
    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);

    void BlockIpAndVisual(Ipv4Address ip);
    void RestoreNodeColor(Ipv4Address ip);
    void UpdateInitialColors();
    void AttackBS(uint32_t bsIndex);
    void RestoreBSColor(uint32_t bsIndex);
};

DDoSSimulator::DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime)
    : m_nIotNodes(nIotNodes), m_nAttackers(nAttackers), m_simTime(simTime) {}

DDoSSimulator::~DDoSSimulator()
{
    if (m_anim) { delete m_anim; m_anim = nullptr; }
}

void DDoSSimulator::CreateNodes()
{
    m_iotNodes.Create(m_nIotNodes);
    m_baseStations.Create(2);
    m_serverNode.Create(1);
    m_allNodes.Add(m_iotNodes);
    m_allNodes.Add(m_baseStations);
    m_allNodes.Add(m_serverNode);

    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    while (m_attackerIndices.size() < m_nAttackers) {
        uint32_t idx = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), idx) == m_attackerIndices.end()) {
            m_attackerIndices.push_back(idx);
        }
    }

    std::cout << "Topology: " << m_nIotNodes << " IoT nodes, "
              << m_nAttackers << " attackers, 2 base stations.\n";
}

void DDoSSimulator::SetupMobility()
{
    // Base stations + server fixed
    Ptr<ListPositionAllocator> infraPos = CreateObject<ListPositionAllocator>();
    infraPos->Add(Vector(30.0, 50.0, 0.0)); // BS1
    infraPos->Add(Vector(70.0, 50.0, 0.0)); // BS2
    infraPos->Add(Vector(50.0, 85.0, 0.0)); // Server
    MobilityHelper staticMob;
    staticMob.SetPositionAllocator(infraPos);
    staticMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    staticMob.Install(m_baseStations);
    staticMob.Install(m_serverNode);

    // IoT nodes rải quanh BS
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        double centerX = (i < m_nIotNodes / 2) ? 30.0 : 70.0;
        double centerY = 50.0;
        double angle = uv->GetValue(0.0, 2*M_PI);
        double radius = uv->GetValue(0.0, 15.0);
        double x = centerX + radius * std::cos(angle);
        double y = centerY + radius * std::sin(angle);

        Ptr<ListPositionAllocator> nodePos = CreateObject<ListPositionAllocator>();
        nodePos->Add(Vector(x, y, 0.0));
        MobilityHelper nodeMob;
        nodeMob.SetPositionAllocator(nodePos);
        nodeMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                 "Bounds", RectangleValue(Rectangle(0, 100, 0, 100)),
                                 "Distance", DoubleValue(3.0));
        nodeMob.Install(m_iotNodes.Get(i));
    }
}

void DDoSSimulator::SetupNetwork()
{
    WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211n);
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy; wifiPhy.SetChannel(wifiChannel.Create());
    WifiMacHelper wifiMac;

    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("ddos-net")));
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);

    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-net")));
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

void DDoSSimulator::SetupApplications()
{
    uint16_t serverPort = 8080;
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), serverPort));
    ApplicationContainer sinkApp = sink.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(m_simTime));

    for (uint32_t i = 0; i < m_nIotNodes; ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end()) continue;
        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, serverPort));
        onOff.SetAttribute("DataRate", StringValue("50kbps"));
        onOff.SetAttribute("PacketSize", UintegerValue(128));
        onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
        ApplicationContainer app = onOff.Install(m_iotNodes.Get(i));
        app.Start(Seconds(1.0 + (double)i * 0.1));
        app.Stop(Seconds(m_simTime - 1.0));
    }
}

void DDoSSimulator::SetupDDoSAttack()
{
    uint16_t attackPort = 8080;
    for (uint32_t idx : m_attackerIndices) {
        OnOffHelper attack("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, attackPort));
        attack.SetAttribute("DataRate", StringValue("5000kbps"));
        attack.SetAttribute("PacketSize", UintegerValue(1024));
        attack.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
        attack.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer app = attack.Install(m_iotNodes.Get(idx));
        app.Start(Seconds(1.0));
        app.Stop(Seconds(m_simTime - 1.0));

        Ipv4Address ip = m_staInterfaces.GetAddress(idx);
        m_attackerIps.push_back(ip);
        std::cout << "Attacker mapped: node " << idx << " -> " << ip << std::endl;
    }
}

void DDoSSimulator::SetupFlowMonitor()
{
    m_monitor = m_flowMonHelper.InstallAll();
}

bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    if (protocol == 0x0800) {
        Ptr<Packet> p_copy = packet->Copy();
        Ipv4Header header;
        if (p_copy->PeekHeader(header)) {
            if (m_blockedIps.count(header.GetSource())) return true;
        }
    }
    return false;
}

void DDoSSimulator::SetupNetAnim()
{
    m_anim = new AnimationInterface("animation_only.xml");
    m_anim->EnablePacketMetadata(false);
    m_anim->SetMaxPktsPerTraceFile(300000);
    UpdateInitialColors();
}

void DDoSSimulator::UpdateInitialColors()
{
    if (!m_anim) return;

    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        bool isAtk = std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end();
        Ptr<Node> n = m_iotNodes.Get(i);
        m_anim->UpdateNodeColor(n->GetId(), isAtk ? 255 : 0, 0, isAtk ? 0 : 255);
        m_anim->UpdateNodeDescription(n->GetId(), isAtk ? "Attacker" : "IoT");
        m_anim->UpdateNodeSize(n->GetId(), 1.4, 1.4);
    }

    for (uint32_t i = 0; i < m_baseStations.GetN(); ++i) {
        Ptr<Node> n = m_baseStations.Get(i);
        std::stringstream ss; ss << "Gateway " << (i+1);
        m_anim->UpdateNodeColor(n->GetId(), 0, 0, 255);
        m_anim->UpdateNodeDescription(n->GetId(), ss.str());
        m_anim->UpdateNodeSize(n->GetId(), 4.0, 4.0);
    }

    Ptr<Node> server = m_serverNode.Get(0);
    m_anim->UpdateNodeColor(server->GetId(), 128, 0, 128);
    m_anim->UpdateNodeDescription(server->GetId(), "SERVER");
    m_anim->UpdateNodeSize(server->GetId(), 6.0, 6.0);
}

// Đổi màu BS khi bị tấn công
void DDoSSimulator::AttackBS(uint32_t bsIndex)
{
    Ptr<Node> bsNode = m_baseStations.Get(bsIndex);
    m_anim->UpdateNodeColor(bsNode->GetId(), 255, 0, 0);
    Simulator::Schedule(Seconds(m_mitigationRestoreDelay), &DDoSSimulator::RestoreBSColor, this, bsIndex);
}

void DDoSSimulator::RestoreBSColor(uint32_t bsIndex)
{
    Ptr<Node> bsNode = m_baseStations.Get(bsIndex);
    m_anim->UpdateNodeColor(bsNode->GetId(), 0, 0, 255);
}

void DDoSSimulator::BlockIpAndVisual(Ipv4Address ip)
{
    if (m_blockedIps.insert(ip).second) {
        for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
            if (m_staInterfaces.GetAddress(i) == ip) {
                m_anim->UpdateNodeColor(m_iotNodes.Get(i)->GetId(), 255, 255, 0);
                m_anim->UpdateNodeDescription(m_iotNodes.Get(i)->GetId(), "QUARANTINED");
                Simulator::Schedule(Seconds(m_mitigationRestoreDelay), &DDoSSimulator::RestoreNodeColor, this, ip);
                break;
            }
        }
    }
}

void DDoSSimulator::RestoreNodeColor(Ipv4Address ip)
{
    m_blockedIps.erase(ip);
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        if (m_staInterfaces.GetAddress(i) == ip) {
            m_anim->UpdateNodeColor(m_iotNodes.Get(i)->GetId(), 0, 255, 0);
            m_anim->UpdateNodeDescription(m_iotNodes.Get(i)->GetId(), "IoT (restored)");
            break;
        }
    }
}

void DDoSSimulator::MonitorLiveFlows()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;
    if (!m_monitor) { Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this); return; }

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;
        FlowId id = it->first;
        uint32_t now = it->second.txPackets;
        uint32_t last = m_lastTxPackets_realtime[id];
        uint32_t delta = (now > last) ? now - last : 0;
        m_lastTxPackets_realtime[id] = now;

        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(id);
        Ipv4Address src = t.sourceAddress;

        if (m_blockedIps.count(src)) continue;
        bool isAttacker = std::find(m_attackerIps.begin(), m_attackerIps.end(), src) != m_attackerIps.end();

        if (isAttacker && delta >= m_attackDetectThresholdPacketsPerSec) {
            // đổi màu IoT node
            for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
                if (m_staInterfaces.GetAddress(i) == src) {
                    m_anim->UpdateNodeColor(m_iotNodes.Get(i)->GetId(), 255, 0, 0);
                    m_anim->UpdateNodeDescription(m_iotNodes.Get(i)->GetId(), "UNDER ATTACK");
                    break;
                }
            }
            // attack BS tương ứng
            uint32_t bsIndex = (std::find_if(m_iotNodes.Begin(), m_iotNodes.End(),
                               [src, this](Ptr<Node> n){ return m_staInterfaces.GetAddress(n->GetId()) == src; }) - m_iotNodes.Begin()) < m_nIotNodes/2 ? 0 : 1;
            AttackBS(bsIndex);

            // block IoT node
            Simulator::Schedule(Seconds(0.1), &DDoSSimulator::BlockIpAndVisual, this, src);
        }
    }

    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();

    for (uint32_t i = 0; i < m_apDevices.GetN(); ++i)
        m_apDevices.Get(i)->SetReceiveCallback(MakeCallback(&DDoSSimulator::PacketDropCallback, this));

    SetupFlowMonitor();
    SetupNetAnim();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);

    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();
    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    uint32_t nodes = 20;
    uint32_t attackers = 4;
    double time = 60.0;

    CommandLine cmd;
    cmd.AddValue("nodes", "Number of IoT nodes", nodes);
    cmd.AddValue("attackers", "Number of attacker nodes", attackers);
    cmd.AddValue("time", "Simulation time (s)", time);
    cmd.Parse(argc, argv);

    DDoSSimulator sim(nodes, attackers, time);
    sim.Run();

    return 0;
}
